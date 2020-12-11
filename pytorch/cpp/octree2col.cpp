#include <octree_nn.h>
#include <octree_parser.h>

#include "ocnn.h"

namespace {

class Octree2ColBase {
 public:
  explicit Octree2ColBase(int depth, std::vector<int> kernel_size, int stride)
      : depth_(depth), kernel_size_(kernel_size), stride_(stride) {
    resize_with_last_val(kernel_size_, 3);
    CHECK_GT(depth_, 0) << "Depth should be larger than 0";
    for (auto k : kernel_size_) {
      CHECK(0 < k && k < 4) << "Invalide kernel size";
    }
    CHECK(stride_ == 1 || stride_ == 2) << "Unsupport stride";
  }

  Tensor init_ni_ptr(torch::TensorOptions options) {
    std::vector<int>& ni_cpu = NeighHelper::Get().get_ni(kernel_size_);

    int count = ni_cpu.size();
    Tensor ni_gpu = torch::zeros({count}, options.dtype(torch::kInt32));
    memcpy_gpu(count, ni_cpu.data(), ni_gpu.data_ptr<int>());
    return ni_gpu;
  }

 protected:
  int depth_;
  vector<int> kernel_size_;
  int stride_;
};

class OctreeToColOp : public Octree2ColBase {
 public:
  explicit OctreeToColOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor data_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(data_in.options());
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // input data, data format: [1, channels, H, 1]
    int btm_depth = this->depth_;
    int channel = data_in.size(1);
    int btm_height = data_in.size(2);
    CHECK_EQ(octree_.info().node_num(btm_depth), btm_height);

    // output data
    int top_height = btm_height;
    if (this->stride_ == 2) {
      top_height = btm_height / 8;
      int top_depth = btm_depth - 1;
      CHECK_EQ(top_height, octree_.info().node_num_nempty(top_depth));
    }
    int kernel_sdim = num_elements(this->kernel_size_);
    Tensor data_out = torch::zeros({channel, kernel_sdim, top_height}, data_in.options());

    // execute
    octree2col_gpu(data_out.data_ptr<float>(), data_in.data_ptr<float>(),
                   channel, top_height, kernel_sdim, this->stride_,
                   octree_.neighbor_gpu(btm_depth), ni_ptr, top_height, 0);
    return data_out;
  }
};

class ColToOctreeOp : public Octree2ColBase {
 public:
  explicit ColToOctreeOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor grad_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(grad_in.options());
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // in grad shape, data format: [channel, kernel_sdim, top_height]
    int channel = grad_in.size(0);
    int top_height = grad_in.size(2);

    // out grad
    int btm_depth = this->depth_;
    int btm_height = octree_.info().node_num(btm_depth);
    if (this->stride_ == 2) {
      CHECK_EQ(top_height, octree_.info().node_num_nempty(btm_depth - 1));
    }
    Tensor grad_out = torch::zeros({1, channel, btm_height, 1}, grad_in.options());

    // execute
    int kernel_sdim = num_elements(this->kernel_size_);
    col2octree_gpu(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(),
                   channel, top_height, kernel_sdim, this->stride_,
                   octree_.neighbor_gpu(btm_depth), ni_ptr, top_height, 0);
    return grad_out;
  }
};

} // anonymous namespace

// API implementation
Tensor octree2col(Tensor data_in, Tensor octree, int depth,
                  std::vector<int> kernel_size, int stride) {
  OctreeToColOp op(depth, kernel_size, stride);
  return op.compute(data_in, octree);
}

Tensor col2octree(Tensor grad_in, Tensor octree, int depth,
                  std::vector<int> kernel_size, int stride) {
  ColToOctreeOp op(depth, kernel_size, stride);
  return op.compute(grad_in, octree);
}
