#ifndef SPATIALQUERYBENCHMARK_LOADER_H
#define SPATIALQUERYBENCHMARK_LOADER_H
#include <dirent.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/box.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <vector>

#include "input_type.h"
#include "utils/type_traits.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
std::vector<typename cuda_vec<COORD_T, N_DIMS>::type> LoadWKTPoints(
    const std::string& path, int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  using boost_point_t =
      boost::geometry::model::point<COORD_T, N_DIMS,
                                    boost::geometry::cs::cartesian>;
  using boost_polygon_t = boost::geometry::model::polygon<boost_point_t>;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points;

  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      if (line.rfind("MULTIPOLYGON", 0) == 0) {
        boost::geometry::model::multi_polygon<boost_polygon_t> multi_poly;
        boost::geometry::read_wkt(line, multi_poly);

        for (auto& poly : multi_poly) {
          for (auto& p : poly.outer()) {
            points.push_back(*reinterpret_cast<point_t*>(&p));
            if (points.size() >= limit) {
              break;
            }
          }
        }
      } else if (line.rfind("POLYGON", 0) == 0) {
        boost_polygon_t poly;
        boost::geometry::read_wkt(line, poly);

        for (auto& p : poly.outer()) {
          points.push_back(*reinterpret_cast<point_t*>(&p));
          if (points.size() >= limit) {
            break;
          }
        }
      } else if (line.rfind("POINT", 0) == 0) {
        boost_point_t p;
        boost::geometry::read_wkt(line, p);

        points.push_back(*reinterpret_cast<point_t*>(&p));
        if (points.size() >= limit) {
          break;
        }
      } else {
        std::cerr << "Bad Geometry " << line << "\n";
        abort();
      }
    }
  }
  ifs.close();
  return points;
}

template <typename COORD_T, int N_DIMS>
std::vector<typename cuda_vec<COORD_T, N_DIMS>::type> LoadOFFPoints(
    const std::string& path, int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points;
  size_t numVertices = 0, numFaces = 0, numEdges = 0;

  CHECK(ifs.is_open()) << "Could not open file " << path;

  std::getline(ifs, line);

  CHECK(line.substr(0, 3) == "OFF")
      << "Not a valid OFF file. First token must be 'OFF'.";

  line = line.substr(3);

  if (line.empty()) {
    std::getline(ifs, line);
  }
  std::istringstream headerStream(line);

  // Read the counts either from the same line or next line
  headerStream >> numVertices >> numFaces >> numEdges;

  CHECK_GT(numVertices, 0) << "Vertex count is zero or not found.";

  for (size_t i = 0; i < numVertices; ++i) {
    COORD_T coord;
    point_t p;

    for (int dim = 0; dim < N_DIMS; ++dim) {
      ifs >> coord;
      reinterpret_cast<COORD_T*>(&p.x)[dim] = coord;
    }
    points.push_back(p);
    if (points.size() >= limit) {
      break;
    }
  }

  ifs.close();
  return points;
}

template <typename POINT_T>
void SerializePoints(const char* file, const std::vector<POINT_T>& points) {
  std::ofstream ofs(file, std::ios::binary);

  size_t size = points.size();
  ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));

  // Write the vector data
  if (size > 0) {
    ofs.write(reinterpret_cast<const char*>(points.data()),
              sizeof(POINT_T) * size);
  }

  ofs.close();
}

template <typename POINT_T>
std::vector<POINT_T> DeserializePoints(const char* file) {
  std::vector<POINT_T> deserialized_points;
  std::ifstream ifs(file, std::ios::binary);
  // Read the size first
  size_t size = 0;
  ifs.read(reinterpret_cast<char*>(&size), sizeof(size));

  // Resize the vector to hold the data
  deserialized_points.resize(size);

  // Read the vector data
  if (size > 0) {
    ifs.read(reinterpret_cast<char*>(deserialized_points.data()),
             sizeof(POINT_T) * size);
  }

  ifs.close();
  return deserialized_points;
}

template <typename COORD_T, int N_DIMS>
std::vector<typename cuda_vec<COORD_T, N_DIMS>::type> LoadPoints(
    const std::string& path, const std::string& serialize_prefix,
    InputType input_type, int limit = std::numeric_limits<int>::max()) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::string escaped_path;
  std::replace_copy(path.begin(), path.end(), std::back_inserter(escaped_path),
                    '/', '_');

  if (!serialize_prefix.empty()) {
    DIR* dir = opendir(serialize_prefix.c_str());
    if (dir) {
      closedir(dir);
    } else if (ENOENT == errno) {
      if (mkdir(serialize_prefix.c_str(), 0755)) {
        std::cerr << "Cannot create dir " << path;
        abort();
      }
    } else {
      std::cerr << "Cannot open dir " << path;
      abort();
    }
  }
  auto ser_path = serialize_prefix + "/points_" + escaped_path + "_limit_" +
                  std::to_string(limit) + ".bin";

  std::vector<point_t> points;

  if (access(ser_path.c_str(), R_OK) == 0) {
    points = DeserializePoints<point_t>(ser_path.c_str());
  } else {
    switch (input_type) {
    case InputType::kWKT: {
      points = LoadWKTPoints<COORD_T, N_DIMS>(path, limit);
      break;
    }
    case InputType::kOFF: {
      points = LoadOFFPoints<COORD_T, N_DIMS>(path, limit);
      break;
    }
    }

    if (!points.empty() && !serialize_prefix.empty()) {
      SerializePoints<point_t>(ser_path.c_str(), points);
    }
  }
  return points;
}

}  // namespace hd

#endif  // SPATIALQUERYBENCHMARK_LOADER_H
