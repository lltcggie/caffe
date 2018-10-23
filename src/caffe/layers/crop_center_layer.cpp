#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_center_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void CropCenterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  const CropCenterParameter& param = this->layer_param_.crop_center_param();
  CHECK_EQ(bottom.size(), 1) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  CHECK_EQ(param.crop_size().size(), input_dim) << "Wrong param.crop_size.";
  for(int i = 0; i < input_dim; i++) {
	int crop_size = param.crop_size(i);
	CHECK_LE(crop_size * 2 + 1, bottom[0]->shape(i)) << "crop size bigger than input shape";
  }
}

template <typename Dtype>
void CropCenterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CropCenterParameter& param = this->layer_param_.crop_center_param();
  int input_dim = bottom[0]->num_axes();

  // Initialize crop_sizes_ to 0 and the new shape to the current shape of the data.
  vector<int> new_shape(bottom[0]->shape());
  vector<int> offsets_shape(1, input_dim);
  crop_sizes_.Reshape(offsets_shape);
  int* crop_size_data = crop_sizes_.mutable_cpu_data();

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < input_dim; ++i) {
	int crop_size = param.crop_size(i);
    int new_size = bottom[0]->shape(i) - crop_size * 2;
    new_shape[i] = new_size;
	crop_size_data[i] = crop_size;
  }
  top[0]->Reshape(new_shape);
  // Compute strides
  src_strides_.Reshape(offsets_shape);
  dest_strides_.Reshape(offsets_shape);
  for (int i = 0; i < input_dim; ++i) {
    src_strides_.mutable_cpu_data()[i] = bottom[0]->count(i + 1, input_dim);
    dest_strides_.mutable_cpu_data()[i] = top[0]->count(i + 1, input_dim);
  }
}

template <typename Dtype>
void CropCenterLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const int* crop_sizes,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  int crop_size = crop_sizes[cur_dim];
  if (cur_dim + 1 < bottom[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursively
    for (int i = crop_size; i < bottom[0]->shape(cur_dim) - crop_size; ++i) {
      indices[cur_dim] = i;
      crop_copy(bottom, top, crop_sizes, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    std::vector<int> ind_red(cur_dim + 1, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);
    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j] - crop_sizes[j];
      ind_off[j] = indices[j];
    }
	ind_red[cur_dim] = 0;
	ind_off[cur_dim] = crop_sizes[cur_dim];
    // do the copy
	int N = top[0]->shape(cur_dim);
    if (is_forward) {
      caffe_copy(N,
          src_data + bottom[0]->offset(ind_off),
          dest_data + top[0]->offset(ind_red));
    } else {
      // in the backwards pass the src_data is top_diff
      // and the dest_data is bottom_diff
      caffe_copy(N,
          src_data + top[0]->offset(ind_red),
          dest_data + bottom[0]->offset(ind_off));
    }
  }
}

template <typename Dtype>
void CropCenterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  crop_copy(bottom, top, crop_sizes_.cpu_data(), indices, 0, bottom_data, top_data,
      true);
}

template <typename Dtype>
void CropCenterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    std::vector<int> indices(top[0]->num_axes(), 0);
    crop_copy(bottom, top, crop_sizes_.cpu_data(), indices, 0, top_diff,
        bottom_diff, false);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CropCenterLayer);
#endif

INSTANTIATE_CLASS(CropCenterLayer);
REGISTER_LAYER_CLASS(CropCenter);

}  // namespace caffe
