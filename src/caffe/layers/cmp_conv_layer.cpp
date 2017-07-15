#include <vector>
#include <iostream>
#include "caffe/kmeans.hpp"
using namespace std;
#include "caffe/layers/cmp_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::ComputeBlobMask()
{
  //  LOG(INFO)<<"conv blob mask"<<endl;
  int count = this->blobs_[0]->count();
  //this->masks_.resize(count);

  //this->dmasks_ = new Dtype[count] ;

  //this->indices_.resize(count);
  //this->centroids_.resize(this->class_num_);

  //calculate min max value of weight
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype min_weight = weight[0] , max_weight = weight[0];
  vector<Dtype> sort_weight(count);
               
  for (int i = 0; i < count; ++i)
  {
    //this->masks_[i] = 1; //initialize
     sort_weight[i] = fabs(weight[i]);
  }

  sort(sort_weight.begin(), sort_weight.end());
  
  max_weight = sort_weight[count - 1];
  
  float ratio = this->sparse_ratio_;

  int index = int(count*ratio) ; //int(count*(1- max_weight)) ;
  Dtype thr ;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  float rat = 0;
  if(index > 0){

    //thr = ratio;
    thr= sort_weight[index-1];
    LOG(INFO) << "CONV THR: " <<thr << " " <<ratio <<endl;


    for (int i = 0; i < count; ++i)
    {
	this->masks_[i] = ((weight[i] > thr || weight[i] < -thr) ? 1 : 0) ;
        //this->masks_[i] = (weight[i] > thr ? 1 : 0);//(weight[i]==0 ? 0 :1) ;//(weight[i] > thr ? 1 : 0);
        muweight[i] *= this->masks_[i];
        //this->dmasks_[i] = this->masks_[i] ;
        rat += (1-this->masks_[i]) ;
     }
   
  }
  else {
      for (int i = 0; i < count; ++i)
      {
          this->masks_[i] = (weight[i]==0 ? 0 :1); //keep unchanged
	  rat += (1-this->masks_[i]) ;
      }
  }
   LOG(INFO) << "sparsity: "<< rat/count <<endl;
  //min_weight = sort_weight[index];
  if(this->quantize_term_)
  {
    int nCentroid = this->class_num_;
    kmeans_cluster(this->indices_, this->centroids_, muweight, count, this->masks_,/* max_weight, min_weight,*/ nCentroid, 1000);
  }
}
template <typename Dtype>
void CmpConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  int count = this->blobs_[0]->count();
  //vector<Dtype> sort_weight(count);
  for (int i = 0; i < count; ++i)
    muweight[i] *= this->masks_[i] ;

  if(this->quantize_term_)
  {
    for (int i = 0; i < count; ++i)
    {
       if (this->masks_[i])
         muweight[i] = this->centroids_[this->indices_[i]];
    }
  }
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 LOG(INFO) << "conv backward" << endl;
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        for (int j = 0; j < count; ++j)
          weight_diff[j] *= this->masks_[j];
        if(this->quantize_term_)
        {
	  vector<Dtype> tmpDiff(this->class_num_);
          vector<int> freq(this->class_num_);
          for (int j = 0; j < count; ++j)
          {
            if (this->masks_[j])
            {
              tmpDiff[this->indices_[j]] += weight_diff[j];
              freq[this->indices_[j]]++;
            }
          }
          for (int j = 0; j < count; ++j)
          {
            if (this->masks_[j])
              weight_diff[j] = tmpDiff[this->indices_[j]] / freq[this->indices_[j]];
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CmpConvolutionLayer);
#endif

INSTANTIATE_CLASS(CmpConvolutionLayer);

}  // namespace caffe
