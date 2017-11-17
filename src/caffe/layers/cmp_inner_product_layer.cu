#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cmp_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {
template <typename Dtype>
__global__ void mask_weight(int n, const Dtype* weight, const int* mask, Dtype* out
  ) { 
  CUDA_KERNEL_LOOP(index, n) {
    
    out[index] = weight[index] * mask[index];
  }
}

template <typename Dtype>
__global__ void quantize_weight_forward( int n, const int* mask, const int* indice, const Dtype* centroid, Dtype* out 
  ) { 
  CUDA_KERNEL_LOOP(index, n) {

    if (mask[index])
       out[index] = centroid[indice[index]];

  }
}

template <typename Dtype>
__global__ void quantize_weight_backward( int n, int class_num, const Dtype* diff, const int* mask, const int* indice, Dtype *tmpDiff, int *freq,  Dtype* out                
    
  ) { 
    
  CUDA_KERNEL_LOOP(index, n) {
  
          tmpDiff[index] = 0;
          freq[index] = 0;
  }
  CUDA_KERNEL_LOOP(index, n) {

          if (mask[index])
          {
               tmpDiff[indice[index]] += diff[index];
               freq[indice[index]]++;
          }
  }

  CUDA_KERNEL_LOOP(index, n){

          if (mask[index])
          {
              out[index] = tmpDiff[indice[index]]/(freq[indice[index]] + 1e-6) ;
          }

  }
}

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = this->blobs_[0]->count();
  mask_weight<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,this->blobs_[0]->gpu_data() ,this->masks_.gpu_data(),this->blobs_[0]->mutable_gpu_data());

  if(this->quantize_term_)
  {
    quantize_weight_forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->masks_.gpu_data(), this->indices_.gpu_data(), this->centroids_.gpu_data(), this->blobs_[0]->mutable_gpu_data());

  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    int count = this->blobs_[0]->count();
  if (this->param_propagate_down_[0]) {

    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }

    mask_weight<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,this->blobs_[0]->gpu_diff() ,this->masks_.gpu_data(),this->blobs_[0]->mutable_gpu_diff());

    if(this->quantize_term_)    
    {
	quantize_weight_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count, this->class_num_, this->blobs_[0]->gpu_diff(), this->masks_.gpu_data(), this->indices_.gpu_data(), this->tmpDiff_.mutable_gpu_data(), this->freq_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());

    }
   }
  
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CmpInnerProductLayer);

}  // namespace caffe
