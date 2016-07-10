#ifndef CAFFE_UTIL_CUDNN_FUNC_H_
#define CAFFE_UTIL_CUDNN_FUNC_H_

#ifdef USE_CUDNN

#include "caffe/util/cudnn.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

namespace cudnn {

template <typename Dtype>
cudnnStatus_t FindConvolutionForwardAlgorithmEx(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<Dtype>*const*            bottom,
	const cudnnFilterDescriptor_t       filter_desc,
	const Blob<Dtype>*                  blob,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       top_descs[],
	Blob<Dtype>*const*                  top,
	cudnnConvolutionFwdAlgo_t           algos[],
	int                                 count);

template <typename Dtype>
cudnnStatus_t FindConvolutionBackwardFilterAlgorithmEx(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<Dtype>*const*            bottom,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<Dtype>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnFilterDescriptor_t       filter_desc,
	int                                 weights_size,
	cudnnConvolutionBwdFilterAlgo_t     algos[],
	int                                 count);

template <typename Dtype>
cudnnStatus_t FindConvolutionBackwardDataAlgorithmEx(
	cudnnHandle_t                       handle[],
	const cudnnFilterDescriptor_t       filter_desc,
	const Blob<Dtype>*                  blob,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<Dtype>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	Blob<Dtype>*const*                  bottom,
	cudnnConvolutionBwdDataAlgo_t       algos[],
	int                                 count);

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_FUNC_H_
