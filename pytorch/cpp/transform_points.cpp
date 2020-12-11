#include <math_functions.h>
#include <octree_nn.h>
#include <points.h>

#include "ocnn.h"

vector<float> bounding_sphere(Tensor data_in, string method) {
  // init the points
  Points pts;
  pts.set(data_in.data_ptr<uint8_t>());

  // check the points
  string msg;
  bool succ = pts.info().check_format(msg);
  if (!succ) return {0.0f};

  // bounding sphere
  float radius = 0.0f, center[3] = {0.0f};
  if (method == "sphere") {
    bounding_sphere(radius, center, pts.points(), pts.info().pt_num());
  } else {
    float bbmin[3] = {0.0f}, bbmax[3] = {0.0f};
    bounding_box(bbmin, bbmax, pts.points(), pts.info().pt_num());
    for (int j = 0; j < 3; ++j) {
      center[j] = (bbmax[j] + bbmin[j]) / 2.0f;
      float width = (bbmax[j] - bbmin[j]) / 2.0f;
      radius += width * width;
    }
    radius = sqrtf(radius + 1.0e-20f);
  }

  // outputs
  return {radius, center[0], center[1], center[2]};
}

namespace {
void setup_transform(Tensor data_in, Tensor& data_out, Points& pts) {
  data_out = torch::zeros_like(data_in);
  uint8_t* out_ptr = data_out.data<uint8_t>();
  memcpy(out_ptr, data_in.data_ptr<uint8_t>(), data_out.numel());

  // init and checkthe points
  pts.set(out_ptr);
  string msg;
  bool succ = pts.info().check_format(msg);
  CHECK(succ) << msg;
}
}  // anonymous namespace

Tensor normalize_points(Tensor data_in, float radius, vector<float> center) {
  // setup
  Tensor data_out;  Points pts;
  setup_transform(data_in, data_out, pts);

  // centralize
  const float dis[3] = {-center[0], -center[1], -center[2]};
  if (dis[0] != 0.0f || dis[1] != 0.0f || dis[2] != 0.0f) {
    pts.translate(dis);
  }

  // scale to [-1, 1]
  CHECK_GE(radius, 0.0f);
  const float inv_radius = 1.0f / radius;
  const float scale[3] = {inv_radius, inv_radius, inv_radius};
  if (scale[0] != 1.0f || scale[1] != 1.0f || scale[2] != 1.0f) {
    pts.scale(scale);
  }

  return data_out;
}

Tensor transform_points(Tensor data_in, vector<float> angle, vector<float> scale, 
                        vector<float> jitter, float offset) {
  // copy the data out of the input tensor
  Tensor data_out;  Points pts;
  setup_transform(data_in, data_out, pts);

  // displacement
  const float kEPS = 1.0e-10f;
  if (offset > kEPS) {
    pts.displace(offset);
  }

  // data augmentation: rotate the point cloud
  resize_with_last_val(angle, 3);
  if (fabs(angle[0]) > kEPS || fabs(angle[1]) > kEPS || fabs(angle[2]) > kEPS) {
    pts.rotate(angle.data());
  }

  // jitter
  resize_with_last_val(jitter, 3);
  if (fabs(jitter[0]) > kEPS || fabs(jitter[1]) > kEPS || fabs(jitter[2]) > kEPS) {
    pts.translate(jitter.data());
  }

  // scale
  resize_with_last_val(scale, 3);
  if (scale[0] != 1.0f || scale[1] != 1.0f || scale[2] != 1.0f) {
    pts.scale(scale.data());
  }

  // clip the points to the box[-1, 1] ^ 3,
  const float bbmin[] = {-1.0f, -1.0f, -1.0f};
  const float bbmax[] = {1.0f, 1.0f, 1.0f};
  pts.clip(bbmin, bbmax);

  // output
  return data_out;
}
