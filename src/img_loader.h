
#ifndef IMG_LOADER_H
#define IMG_LOADER_H
#include <string>
#include <vector>

#include "utils/type_traits.h"

template <typename COORD_T, int N_DIMS>
std::vector<typename cuda_vec<COORD_T, N_DIMS>::type> LoadImage(
    const std::string& path, int limit = std::numeric_limits<int>::max()) {
  using PixelType = unsigned char;
  using ImageType = itk::Image<PixelType, N_DIMS>;
  using ReaderType = itk::ImageFileReader<ImageType>;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points;

  auto reader = ReaderType::New();
  reader->SetFileName(path);
  try {
    reader->Update();
  } catch (itk::ExceptionObject& err) {
    std::cerr << "Error: " << err << std::endl;
    return {};
  }
  auto image = reader->GetOutput();
  // Iterator over the image
  using IteratorType = itk::ImageRegionIterator<ImageType>;
  IteratorType it(image, image->GetLargestPossibleRegion());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    PixelType value = it.Get();
    if (value > 0) {  // Non-empty voxel
      auto index = it.GetIndex();
      point_t p;

      for (int dim = 0; dim < N_DIMS; ++dim) {
        reinterpret_cast<COORD_T*>(&p)[dim] = index[dim];
      }
      points.push_back(p);
      if (points.size() >= limit) {
        break;
      }
    }
  }
  return points;
}

#endif  // IMG_LOADER_H
