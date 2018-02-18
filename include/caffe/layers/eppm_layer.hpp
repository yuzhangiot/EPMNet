#ifndef EPPM_LAYER_HPP_
#define EPPM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief EPPM patches from the first input image with patches from
 *        the second input image with bilateral weight
 *
 */
template <typename Dtype>
class EppmLayer : public Layer<Dtype> {
 public:

  explicit EppmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Eppm"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_size_;
  
  int stride1_;
  int stride2_;
  int max_displacement_;
  
  int pad_size_;
  
  int num_;
  int top_height_, top_width_;
  int top_channels_;
  
  // EPPM specific:
  bool do_abs_;
  EppmParameter_EppmType eppm_type_;
  
  shared_ptr< Blob<Dtype> > rbot1_;
  shared_ptr< Blob<Dtype> > rbot2_;
  
  shared_ptr< Blob<Dtype> > rtopdiff_;
  
  // Computed: (For explanation, see Reshape and Forward functions)
  int kernel_radius_;
  int border_size_;
  int neighborhood_grid_radius_, neighborhood_grid_width_;

};


}  // namespace caffe

#endif  // EPPM_LAYER_HPP_
