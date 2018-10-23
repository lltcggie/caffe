#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/crop_center_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CropCenterLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CropCenterLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 5, 7)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);

    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CropCenterLayerTest() {
    delete blob_bottom_0_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(CropCenterLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropCenterLayerTest, TestSetupShapeAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Crop all dimensions
  auto& crop_size = *layer_param.mutable_crop_center_param()->mutable_crop_size();
  crop_size.Add(0);
  crop_size.Add(1);
  crop_size.Add(1);
  crop_size.Add(2);
  CropCenterLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->num_axes(); ++i) {
    EXPECT_EQ(this->blob_bottom_0_->shape(i) - crop_size[i] * 2, this->blob_top_->shape(i));
  }
}

TYPED_TEST(CropCenterLayerTest, TestCropAll) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  auto& crop_size = *layer_param.mutable_crop_center_param()->mutable_crop_size();
  crop_size.Add(0);
  crop_size.Add(1);
  crop_size.Add(1);
  crop_size.Add(2);
  CropCenterLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = crop_size[0]; n < this->blob_bottom_0_->num() - crop_size[0]; ++n) {
    for (int c = crop_size[1]; c < this->blob_bottom_0_->channels() - crop_size[1]; ++c) {
      for (int h = crop_size[2]; h < this->blob_bottom_0_->height() - crop_size[2]; ++h) {
        for (int w = crop_size[3]; w < this->blob_bottom_0_->width() - crop_size[3]; ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n - crop_size[0], c - crop_size[1], h - crop_size[2], w - crop_size[3]),
              this->blob_bottom_0_->data_at(n, c, h, w));
        }
      }
    }
  }
}

}  // namespace caffe
