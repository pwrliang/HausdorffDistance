#ifndef HAUSDORFF_DISTANCE_ITK_H
#define HAUSDORFF_DISTANCE_ITK_H
#include <glog/logging.h>

#include "hausdorff_distance.h"
#include "itkDirectedHausdorffDistanceImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceITK : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using PixelType = unsigned char;
  using ImageType = itk::Image<PixelType, N_DIMS>;

 public:
  struct Config {
    int n_threads;
    itk::Size<N_DIMS> size_a;
    itk::Size<N_DIMS> size_b;
  };

  HausdorffDistanceITK() = default;

  explicit HausdorffDistanceITK(const Config& config) : config_(config) {
    CHECK_GT(config.n_threads, 0);
  }

  coord_t CalculateDistance(std::vector<point_t>& points_a,
                            std::vector<point_t>& points_b) override {
    Stopwatch sw;
    sw.start();
    auto& stats = this->stats_;
    auto image_a = CreateImage(points_a, config_.size_a);
    auto image_b = CreateImage(points_b, config_.size_b);
    sw.stop();
    stats["CreateImageTime"] = sw.ms();

    using FilterType =
        itk::DirectedHausdorffDistanceImageFilter<ImageType, ImageType>;

    sw.start();
    auto hausdorffFilter = FilterType::New();
    hausdorffFilter->SetNumberOfWorkUnits(config_.n_threads);
    hausdorffFilter->SetInput1(image_a);
    hausdorffFilter->SetInput2(image_b);
    hausdorffFilter->Update();

    auto hd = hausdorffFilter->GetDirectedHausdorffDistance();
    sw.stop();

    stats["Algorithm"] = "ITK";
    stats["Execution"] = "CPU";
    stats["Threads"] = config_.n_threads;
    stats["ReportedTime"] = sw.ms();
    return hd;
  }

 private:
  itk::SmartPointer<ImageType> CreateImage(const std::vector<point_t>& points,
                                           const itk::Size<N_DIMS>& size) {
    auto image = ImageType::New();

    typename ImageType::RegionType region;
    region.SetSize(size);
    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0);

    typename ImageType::IndexType index;
    for (const auto& p : points) {
      for (int dim = 0; dim < N_DIMS; dim++) {
        index[dim] = static_cast<int>(
            std::round(reinterpret_cast<const COORD_T*>(&p.x)[dim]));
      }
      if (region.IsInside(index)) {
        image->SetPixel(index, 1);
      }
    }

    return image;
  }

  Config config_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_ITK_H
