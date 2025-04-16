// #ifndef HAUSDORFF_DISTANCE_ITK_H
// #define HAUSDORFF_DISTANCE_ITK_H
// #include "itkDirectedHausdorffDistanceImageFilter.h"
// #include "itkImage.h"
// #include "itkImageFileReader.h"
// #include "itkImageFileWriter.h"
// #include "utils/stopwatch.h"
// namespace hd {
// template <int N_DIMS>
// double CalculateHausdorffDistanceITK(const char* input1, const char* input2,
//                                      double& loading_time_ms) {
//   Stopwatch sw;
//   sw.start();
//   using PixelType = unsigned char;
//   using ImageType = itk::Image<PixelType, 3>;
//   using ReaderType = itk::ImageFileReader<ImageType>;
//   typename ReaderType::Pointer sourceReader = ReaderType::New();
//   typename ReaderType::Pointer targetReader = ReaderType::New();
//   sourceReader->SetFileName(input1);  // Input binary image
//   targetReader->SetFileName(input2);  // Input binary image
//   using HausdorffFilterType =
//       itk::DirectedHausdorffDistanceImageFilter<ImageType, ImageType>;
//   typename HausdorffFilterType::Pointer hausdorffFilter =
//       HausdorffFilterType::New();
//
//   try {
//     sourceReader->Update();
//     targetReader->Update();
//   } catch (itk::ExceptionObject& err) {
//     std::cerr << "Error: " << err << std::endl;
//     return EXIT_FAILURE;
//   }
//   hausdorffFilter->SetInput1(sourceReader->GetOutput());
//   hausdorffFilter->SetInput2(targetReader->GetOutput());
//   sw.stop();
//
//   try {
//     hausdorffFilter->Update();
//   } catch (itk::ExceptionObject& err) {
//     std::cerr << "Error: " << err << std::endl;
//     return EXIT_FAILURE;
//   }
//
//   loading_time_ms = sw.ms();
//   double distance = hausdorffFilter->GetDirectedHausdorffDistance();
//   return distance;
// }
// }  // namespace hd
//
// #endif  // HAUSDORFF_DISTANCE_ITK_H
