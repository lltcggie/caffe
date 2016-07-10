#ifdef USE_CUDNN
#include "caffe/blob.hpp"
#include "caffe/util/cudnn_func.hpp"
#include <functional>

namespace caffe {
namespace cudnn {

const size_t AlgoMax = 16;

template <typename Perf_t, typename Algo_t>
cudnnStatus_t FindAlgorithmEx(
	std::function<cudnnStatus_t(size_t i, Perf_t * algo, size_t algoSize, int *count, void *workspace, size_t workspaceSize, void *weights)> func,
	Algo_t algos[], int num, int weights_size = -1)
{
	cudnnStatus_t ret;

	void *weights = nullptr;
	if (weights_size > 0)
		CUDA_CHECK(cudaMalloc(&weights, weights_size));

	size_t lastWorkSpace = 0;
	std::vector<Perf_t> algo(AlgoMax);
	int count = 0;
	for (size_t i = 0; i < num; i++)
	{
		ret = func(i, algo.data(), algo.size(), &count, nullptr, 0, weights);
		if (ret != CUDNN_STATUS_SUCCESS)
		{
			if (weights)
				cudaFree(weights);
			return ret;
		}

		size_t free = 0, total = 0;
		CUDA_CHECK(cudaMemGetInfo(&free, &total));

		const size_t limit = free * 0.9;

		size_t maxWorkSpace = 0;
		for (size_t c = 0; c < count; c++)
		{
			const auto &elm = algo[c];
			if (elm.status == CUDNN_STATUS_SUCCESS || elm.status == CUDNN_STATUS_ALLOC_FAILED)
			{
				maxWorkSpace = std::max(maxWorkSpace, std::min(elm.memory, limit));
			}
		}

		void *workspaceData = nullptr;
		if (cudaMalloc(&workspaceData, maxWorkSpace) != cudaSuccess)
		{
			const size_t BaseWorkSpace = maxWorkSpace;

			cudaError_t e;
			// 確保できなかったらどんどん使用メモリを減らして確保してみる
			for (int i = 8; i >= 0; i--)
			{
				maxWorkSpace = BaseWorkSpace * i / 10;
				e = cudaMalloc(&workspaceData, maxWorkSpace);
				if (e == cudaSuccess)
					break;
			}

			CUDA_CHECK(e);
		}

		ret = func(i, algo.data(), algo.size(), &count, workspaceData, maxWorkSpace, weights);

		if (weights)
			cudaFree(weights);

		if (workspaceData)
			cudaFree(workspaceData);

		if (ret != CUDNN_STATUS_SUCCESS)
			return ret;

		algos[i] = algo[0].algo;

		lastWorkSpace = std::max(lastWorkSpace, algo[0].memory);
	}

	return CUDNN_STATUS_SUCCESS;
}

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
	int                                 count)
{
	auto func = [&](size_t i, cudnnConvolutionFwdAlgoPerf_t * algo, size_t algoSize, int *count, void *workspace, size_t workspaceSize, void *weights)
	{
		return cudnnFindConvolutionForwardAlgorithmEx(handle[i],
			bottom_descs[i],
			bottom[i]->gpu_data(),
			filter_desc,
			blob->gpu_data(),
			conv_descs[i],
			top_descs[i],
			top[i]->mutable_gpu_data(),
			algoSize,
			count,
			algo, workspace, workspaceSize);
	};

	return FindAlgorithmEx<cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t>(func, algos, count);
}

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
	int                                 count)
{
	auto func = [&](size_t i, cudnnConvolutionBwdFilterAlgoPerf_t * algo, size_t algoSize, int *count, void *workspace, size_t workspaceSize, void *weights)
	{
		return cudnnFindConvolutionBackwardFilterAlgorithmEx(handle[i],
			bottom_descs[i],
			bottom[i]->gpu_data(),
			top_descs[i],
			top[i]->gpu_diff(),
			conv_descs[i],
			filter_desc,
			weights,
			algoSize,
			count,
			algo, workspace, workspaceSize);
	};

	return FindAlgorithmEx<cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t>(func, algos, count, weights_size);
}

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
	int                                 count)
{
	auto func = [&](size_t i, cudnnConvolutionBwdDataAlgoPerf_t * algo, size_t algoSize, int *count, void *workspace, size_t workspaceSize, void *weights)
	{
		return cudnnFindConvolutionBackwardDataAlgorithmEx(handle[i],
			filter_desc,
			blob->gpu_data(),
			top_descs[i],
			top[i]->gpu_diff(),
			conv_descs[i],
			bottom_descs[i],
			bottom[i]->mutable_gpu_diff(),
			algoSize,
			count,
			algo, workspace, workspaceSize);
	};

	return FindAlgorithmEx<cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t>(func, algos, count);
}


template
cudnnStatus_t FindConvolutionForwardAlgorithmEx<float>(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<float>*const*            bottom,
	const cudnnFilterDescriptor_t       filter_desc,
    const Blob<float>*                  blob,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       top_descs[],
	Blob<float>*const*                  top,
	cudnnConvolutionFwdAlgo_t           algos[],
	int                                 count);

template
cudnnStatus_t FindConvolutionForwardAlgorithmEx<double>(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<double>*const*            bottom,
	const cudnnFilterDescriptor_t       filter_desc,
	const Blob<double>*                  blob,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       top_descs[],
	Blob<double>*const*                  top,
	cudnnConvolutionFwdAlgo_t           algos[],
	int                                 count);

template
cudnnStatus_t FindConvolutionBackwardFilterAlgorithmEx<float>(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<float>*const*            bottom,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<float>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnFilterDescriptor_t       filter_desc,
	int                                 weights_size,
	cudnnConvolutionBwdFilterAlgo_t     algos[],
	int                                 count);

template
cudnnStatus_t FindConvolutionBackwardFilterAlgorithmEx<double>(
	cudnnHandle_t                       handle[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	const Blob<double>*const*            bottom,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<double>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnFilterDescriptor_t       filter_desc,
	int                                 weights_size,
	cudnnConvolutionBwdFilterAlgo_t     algos[],
	int                                 count);

template
cudnnStatus_t FindConvolutionBackwardDataAlgorithmEx<float>(
	cudnnHandle_t                       handle[],
	const cudnnFilterDescriptor_t       filter_desc,
	const Blob<float>*                  blob,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<float>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	Blob<float>*const*                  bottom,
	cudnnConvolutionBwdDataAlgo_t       algos[],
	int                                 count);

template
cudnnStatus_t FindConvolutionBackwardDataAlgorithmEx<double>(
	cudnnHandle_t                       handle[],
	const cudnnFilterDescriptor_t       filter_desc,
	const Blob<double>*                 blob,
	const cudnnTensorDescriptor_t       top_descs[],
	const Blob<double>*const*            top,
	const cudnnConvolutionDescriptor_t  conv_descs[],
	const cudnnTensorDescriptor_t       bottom_descs[],
	Blob<double>*const*                  bottom,
	cudnnConvolutionBwdDataAlgo_t       algos[],
	int                                 count);

}  // namespace cudnn
}  // namespace caffe
#endif
