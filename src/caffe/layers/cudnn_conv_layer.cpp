#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  kernel_w_ = kernel_w;
  kernel_h_ = kernel_h;

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  prev_num_ = 0;
  prev_channels_ = 0;
  prev_group_ = 0;
  prev_num_output_ = 0;
  prev_out_spatial_dim_ = 0;
  prev_width_ = 0;
  prev_width_out_ = 0;
  prev_height_ = 0;
  prev_height_out_ = 0;
  prev_pad_w_ = 0;
  prev_stride_w_ = 0;
  prev_stride_h_ = 0;

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  if (prev_num_ != this->num_ || prev_channels_ != this->channels_ || prev_group_ != this->group_ || prev_num_output_ != this->num_output_ ||
	  prev_out_spatial_dim_ != this->out_spatial_dim_ || prev_width_ != width || prev_width_out_ != width_out || prev_height_ != height ||
	  prev_height_out_ != height_out || prev_pad_w_ != pad_w || prev_pad_h_ != pad_h || prev_stride_w_ != stride_w || prev_stride_h_ != stride_h)
  {
	  prev_num_ = this->num_;
	  prev_channels_ = this->channels_;
	  prev_group_ = this->group_;
	  prev_num_output_ = this->num_output_;
	  prev_out_spatial_dim_ = this->out_spatial_dim_;
	  prev_width_ = width;
	  prev_width_out_ = width_out;
	  prev_height_ = height;
	  prev_height_out_ = height_out;
	  prev_pad_w_ = pad_w;
	  prev_pad_h_ = pad_h;
	  prev_stride_w_ = stride_w;
	  prev_stride_h_ = stride_h;

	  // Specify workspace limit for kernels directly until we have a
	  // planning strategy and a rewrite of Caffe's GPU memory mangagement
	  size_t workspace_limit_bytes = 8 * 1024 * 1024;

	  for (int i = 0; i < bottom.size(); i++) {
		  cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
			  this->num_,
			  this->channels_ / this->group_, height, width,
			  this->channels_ * height * width,
			  height * width, width, 1);
		  cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
			  this->num_,
			  this->num_output_ / this->group_, height_out, width_out,
			  this->num_output_ * this->out_spatial_dim_,
			  this->out_spatial_dim_, width_out, 1);
		  cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
			  filter_desc_, pad_h, pad_w,
			  stride_h, stride_w);

		  const int algo = Caffe::GetcuDNNAlgorithm(type(), this->channels_, this->num_output_, this->num_,
			  width, height, kernel_w_, kernel_h_, pad_w, pad_h, stride_w, stride_h);
		  if (algo >= 0)
			  fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)algo;
		  else
		  {
#if 0
			  // choose forward and backward algorithms + workspace(s)
			  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
				  bottom_descs_[i],
				  filter_desc_,
				  conv_descs_[i],
				  top_descs_[i],
				  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
				  workspace_limit_bytes,
				  &fwd_algo_[i]));
#else
			  int count;
			  cudnnConvolutionFwdAlgoPerf_t choosen_algo_perf;
			  // choose forward and backward algorithms + workspace(s)
			  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle_[0],
				  bottom_descs_[i],
				  filter_desc_,
				  conv_descs_[i],
				  top_descs_[i],
				  1,
				  &count,
				  &choosen_algo_perf));

			  fwd_algo_[i] = choosen_algo_perf.algo;
#endif
			  Caffe::SetcuDNNAlgorithm((int)fwd_algo_[i], type(), this->channels_, this->num_output_, this->num_,
				  width, height, kernel_w_, kernel_h_, pad_w, pad_h, stride_w, stride_h);
		  }

		  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
			  bottom_descs_[i],
			  filter_desc_,
			  conv_descs_[i],
			  top_descs_[i],
			  fwd_algo_[i],
			  &(workspace_fwd_sizes_[i])));

		  if (this->phase_ == TRAIN)
		  {
			  // choose backward algorithm for filter
			  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
				  bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
				  CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
				  workspace_limit_bytes, &bwd_filter_algo_[i]));

			  // get workspace for backwards filter algorithm
			  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
				  bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
				  bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

			  // choose backward algo for data
			  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
				  filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
				  CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
				  workspace_limit_bytes, &bwd_data_algo_[i]));

			  // get workspace size
			  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
				  filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
				  bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
		  }
	  }

	  // reduce over all workspace sizes to get a maximum to allocate / reallocate
	  size_t total_workspace_fwd = 0;
	  size_t total_workspace_bwd_data = 0;
	  size_t total_workspace_bwd_filter = 0;

	  for (size_t i = 0; i < bottom.size(); i++) {
		  total_workspace_fwd = std::max(total_workspace_fwd,
			  workspace_fwd_sizes_[i]);
		  total_workspace_bwd_data = std::max(total_workspace_bwd_data,
			  workspace_bwd_data_sizes_[i]);
		  total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
			  workspace_bwd_filter_sizes_[i]);
	  }
	  // get max over all operations
	  size_t max_workspace = std::max(total_workspace_fwd,
		  total_workspace_bwd_data);
	  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
	  // ensure all groups have enough workspace
	  size_t total_max_workspace = max_workspace *
		  (this->group_ * CUDNN_STREAMS_PER_GROUP);

	  // this is the total amount of storage needed over all groups + streams
	  if (total_max_workspace > workspaceSizeInBytes) {
		  DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
		  workspaceSizeInBytes = total_max_workspace;

		  // free the existing workspace and allocate a new (larger) one
		  cudaFree(this->workspaceData);

		  cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
		  if (err != cudaSuccess) {
			  // force zero memory path
			  for (int i = 0; i < bottom.size(); i++) {
				  workspace_fwd_sizes_[i] = 0;
				  workspace_bwd_filter_sizes_[i] = 0;
				  workspace_bwd_data_sizes_[i] = 0;
				  fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
				  bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
				  bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			  }

			  // NULL out all workspace pointers
			  for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
				  workspace[g] = NULL;
			  }
			  // NULL out underlying data
			  workspaceData = NULL;
			  workspaceSizeInBytes = 0;
		  }

		  // if we succeed in the allocation, set pointer aliases for workspaces
		  for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
			  workspace[g] = reinterpret_cast<wchar_t *>(workspaceData)+g*max_workspace;
		  }
	  }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
