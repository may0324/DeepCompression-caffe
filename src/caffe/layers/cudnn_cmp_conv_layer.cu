#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_cmp_conv_layer.hpp"



namespace caffe {

__global__ void sync_cmp_conv_groups() { }

template <typename Dtype>
__global__ void mask_weight( int n, const Dtype* weight, const int* mask, Dtype* out
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
void CuDNNCmpConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int count = this->blobs_[0]->count();
  mask_weight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->blobs_[0]->gpu_data(), this->masks_.gpu_data(), this->blobs_[0]->mutable_gpu_data());


  if(this->quantize_term_)
  {
    quantize_weight_forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->masks_.gpu_data(), this->indices_.gpu_data(), this->centroids_.gpu_data(), this->blobs_[0]->mutable_gpu_data());
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_cmp_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNCmpConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  int count = 0 ;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    count = this->blobs_[0]->count();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));



	mask_weight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, this->blobs_[0]->gpu_diff(),this->masks_.gpu_data() ,this->blobs_[0]->mutable_gpu_diff());


	if(this->quantize_term_)
	{
	  quantize_weight_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count, this->class_num_, this->blobs_[0]->gpu_diff(), this->masks_.gpu_data(), this->indices_.gpu_data(), this->tmpDiff_.mutable_gpu_data(), this->freq_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());
	}

      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_cmp_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNCmpConvolutionLayer);

}  // namespace caffe
#endif
