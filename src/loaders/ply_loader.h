#ifndef PLY_LOADER_H
#define PLY_LOADER_H
#include "rply/rply.h"
#include "utils/type_traits.h"

namespace hd {
// Helper struct to hold the accumulating data
struct VertexReader {
  std::vector<std::array<double, 3>>* out;  // where to push
  double tmp[3];                            // x,y,z buffer
  int n_dims;
  int limit;
};

// This callback is invoked once per vertex property (x, y, or z)
static int vertex_cb(p_ply_argument argument) {
  void* userdata;
  long prop_index;
  // retrieve our struct pointer and the index (0 for x, 1 for y, 2 for z)
  ply_get_argument_user_data(argument, &userdata, &prop_index);
  VertexReader* r = static_cast<VertexReader*>(userdata);
  auto val = ply_get_argument_value(argument);

  // read the value
  r->tmp[prop_index] = val;

  // once we've read z (prop_index==2), we have a full XYZ triplet
  if (prop_index == r->n_dims - 1) {
    r->out->push_back({r->tmp[0], r->tmp[1], r->tmp[2]});
    r->limit--;
  }

  return r->limit > 0;  // return non-zero to continue
}

template <typename COORD_T, int N_DIMS>
std::vector<typename cuda_vec<COORD_T, N_DIMS>::type> LoadPLY(
    const std::string& filename, int limit = std::numeric_limits<int>::max()) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points;

  p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);
  if (!ply)
    return points;
  if (!ply_read_header(ply))
    return points;

  std::vector<std::array<double, 3>> vertices;
  VertexReader reader{&vertices, {0, 0, 0}, N_DIMS, limit};

  std::string dim_names[N_DIMS];

  ply_set_read_cb(ply, "vertex", "x", vertex_cb, &reader, 0);
  ply_set_read_cb(ply, "vertex", "y", vertex_cb, &reader, 1);
  if (N_DIMS == 3) {
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, &reader, 2);
  }

  if (!ply_read(ply))
    return points;

  points.resize(vertices.size());

  std::array<double, N_DIMS> vmin, vmax, range;

  for (int dim = 0; dim < N_DIMS; dim++) {
    vmin[dim] = std::numeric_limits<double>::max();
    vmax[dim] = std::numeric_limits<double>::lowest();
  }

  for (size_t i = 0; i < vertices.size(); i++) {
    for (int dim = 0; dim < N_DIMS; dim++) {
      vmin[dim] = std::min(vmin[dim], vertices[i][dim]);
      vmax[dim] = std::max(vmax[dim], vertices[i][dim]);
    }
  }

  for (int dim = 0; dim < N_DIMS; dim++) {
    range[dim] = vmax[dim] - vmin[dim];
    if (range[dim] == 0) {
      range[dim] = 1;
    }
  }

  for (auto& v : vertices) {
    for (int i = 0; i < N_DIMS; ++i) {
      v[i] = (v[i] - vmin[i]) / range[i];
    }
  }

  for (size_t i = 0; i < vertices.size(); i++) {
    for (int dim = 0; dim < N_DIMS; dim++) {
      reinterpret_cast<COORD_T*>(&points[i].x)[dim] = vertices[i][dim];
    }
  }

  ply_close(ply);
  return points;
}
}  // namespace hd

#endif  // PLY_LOADER_H
