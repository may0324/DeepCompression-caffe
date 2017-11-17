#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

#include "caffe/layers/cmp_conv_layer.hpp"

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
void CmpConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int count = this->blobs_[0]->count();
  mask_weight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->blobs_[0]->gpu_data(),this->masks_.gpu_data(),this->blobs_[0]->mutable_gpu_data());

  if(this->quantize_term_)
  {
	quantize_weight_forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->masks_.gpu_data(), this->indices_.gpu_data(), this->centroids_.gpu_data(), this->blobs_[0]->mutable_gpu_data());

  }

  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //  LOG(INFO) << "conv backward"<<endl;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }

        mask_weight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->blobs_[0]->gpu_diff(),this->masks_.gpu_data(),this->blobs_[0]->mutable_gpu_diff());


        if(this->quantize_term_)
        {
         quantize_weight_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count, this->class_num_, this->blobs_[0]->gpu_diff(), this->masks_.gpu_data(), this->indices_.gpu_data(), this->tmpDiff_.mutable_gpu_data(), this->freq_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());

        }


        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CmpConvolutionLayer);

}  // namespace caffe
