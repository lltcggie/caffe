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
	int                                 count,
	int                                 dev_no);

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
	int                                 count,
	int                                 dev_no);

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
	int                                 count,
	int                                 dev_no);

class cudnnWorkSpace
{
private:
	typedef std::pair<void *, size_t> WorkSpaceType;
	std::vector<WorkSpaceType> mWorkSpaces;

	static cudnnWorkSpace mObj;

private:
	cudnnWorkSpace()
	{}

	inline cudaError_t AllocWorkSpace(WorkSpaceType &ws, size_t size, int dev_no, bool force = false)
	{
		if (ws.second > size && !force)
			return cudaSuccess;

		if (ws.first)
		{
			CUDA_CHECK(cudaFree(ws.first));
			ws.first = nullptr;
			ws.second = 0;
		}

		cudaError_t ret = cudaMalloc(&ws.first, size);
		if(ret == cudaSuccess)
			ws.second = size;

		return ret;
	}

	inline cudaError_t SetLimitWorkSpace(WorkSpaceType &ws, size_t limit, int dev_no)
	{
		if (ws.second <= limit)
			return cudaSuccess;

		const size_t size = std::min(ws.second, limit);
		return AllocWorkSpace(ws, size, dev_no, true);
	}

public:
	~cudnnWorkSpace()
	{
		Destroy();
	}

	static cudnnWorkSpace& Get()
	{
		return mObj;
	}

	inline cudaError_t GetWorkSpace(size_t size, void** ptr, int dev_no = -1)
	{
		if (dev_no < 0)
		{
			CUDA_CHECK(cudaGetDevice(&dev_no));
		}

		if (mWorkSpaces.size() >= dev_no)
		{
			mWorkSpaces.resize(dev_no + 1, std::make_pair<void *, size_t>(nullptr, 0));
		}

		auto &ws = mWorkSpaces[dev_no];
		cudaError_t ret = AllocWorkSpace(ws, size, dev_no);

		if (ptr)
		{
			*ptr = nullptr;
			if (ws.first)
				*ptr = ws.first;
		}

		return ret;
	}

	inline cudaError_t SetLimitWorkSpace(size_t limit, void** ptr, int dev_no = -1)
	{
		if (dev_no < 0)
		{
			CUDA_CHECK(cudaGetDevice(&dev_no));
		}

		if (mWorkSpaces.size() >= dev_no)
		{
			mWorkSpaces.resize(dev_no + 1, std::make_pair<void *, size_t>(nullptr, 0));
		}

		auto &ws = mWorkSpaces[dev_no];
		cudaError_t ret = SetLimitWorkSpace(ws, limit, dev_no);

		if (ptr)
		{
			*ptr = nullptr;
			if (ws.first)
				*ptr = ws.first;
		}

		return ret;
	}

	size_t GetWorkSpaceSize(int dev_no = -1) const
	{
		if (dev_no < 0)
		{
			CUDA_CHECK(cudaGetDevice(&dev_no));
		}

		if (mWorkSpaces.size() >= dev_no)
			return 0;

		auto &ws = mWorkSpaces[dev_no];

		return ws.second;
	}

	inline void Destroy()
	{
		for (size_t i = 0; i < mWorkSpaces.size(); i++)
		{
			auto &ws = mWorkSpaces[i];
			if (ws.first)
			{
				CUDA_CHECK(cudaFree(ws.first));
			}
		}

		mWorkSpaces.clear();
	}
};

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_FUNC_H_
