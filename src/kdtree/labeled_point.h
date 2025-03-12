#pragma once
#include "cukd/cukd-math.h"
#include "cukd/kdtree.h"
#include "utils/type_traits.h"
#include "utils/util.h"

template <int N_DIMS, typename ID_T = uint32_t>
struct LabeledPoint {
  using fpoint_t = typename cuda_vec<float, N_DIMS>::type;
  LabeledPoint() = default;
  DEV_HOST LabeledPoint(ID_T _id, const fpoint_t& _point)
      : id(_id), point(_point) {}
  ID_T id;
  fpoint_t point;
};

template <typename labeled_point_t,
          typename _point_traits =
              cukd::point_traits<typename labeled_point_t::fpoint_t>>
struct labeled_point_traits {
  // ------------------------------------------------------------------
  /* part I : describes the _types_ of d-dimensional point data that
     the tree will be built over */
  // ------------------------------------------------------------------
  using point_t = typename labeled_point_t::fpoint_t;
  using point_traits = _point_traits;

  // ------------------------------------------------------------------
  /* part II : describes the type of _data_ (which can be more than
     just a point).   */
  // ------------------------------------------------------------------

  using data_t = labeled_point_t;

  // ------------------------------------------------------------------
  /* part III : how to extract a point or coordinate from an actual
     data struct */
  // ------------------------------------------------------------------
 private:
  // this doesn't _need_ to be defined in a data_traits, but makes some of
  // the blow code cleaner to read
  using scalar_t = typename point_traits::scalar_t;

 public:
  /*! return a reference to the 'd'th positional coordinate of the
    given node - for the default simple 'data==point' case we can
    simply return a reference to the point itself */
  static inline __both__ const point_t& get_point(const data_t& n) {
    return n.point;
  }

  /*! return the 'd'th positional coordinate of the given node */
  static inline __both__ scalar_t get_coord(const data_t& n, int d) {
    return point_traits::get_coord(get_point(n), d);
  }

  // ------------------------------------------------------------------
  /* part IV : whether the data has a way of storing a split
     dimension for non-round robin paritioning, and if so, how to
     store (for building) and read (for traversing) that split
     dimensional in/from a node */
  // ------------------------------------------------------------------

  /* whether that node type has a field to store an explicit split
     dimension in each node. If not, the k-d tree builder and
     traverse _have_ to use round-robin for split distance;
     otherwise, it will alwyas split the widest dimensoin */
  enum { has_explicit_dim = false };

  /*! !{ just defining this for completeness, get/set_dim should never
    get called for this type because we have set has_explicit_dim
    set to false. note traversal should ONLY ever call this
    function for data_t's that define has_explicit_dim to true */
  static inline __device__ int get_dim(const data_t&) { return -1; }
  static inline __device__ void set_dim(data_t&, int) {}
  /*! @} */
};