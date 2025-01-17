file(REMOVE_RECURSE
  "../bin/hd_exec"
  "../bin/hd_exec.pdb"
  "../bin/ptx/double_shaders_cull_2d.ptx"
  "../bin/ptx/double_shaders_nn_2d.ptx"
  "../bin/ptx/double_shaders_nn_multicast_2d.ptx"
  "../bin/ptx/float_shaders_cull_2d.ptx"
  "../bin/ptx/float_shaders_nn_2d.ptx"
  "../bin/ptx/float_shaders_nn_multicast_2d.ptx"
  "CMakeFiles/hd_exec.dir/cmake_device_link.o"
  "CMakeFiles/hd_exec.dir/flags.cc.o"
  "CMakeFiles/hd_exec.dir/flags.cc.o.d"
  "CMakeFiles/hd_exec.dir/main.cpp.o"
  "CMakeFiles/hd_exec.dir/main.cpp.o.d"
  "CMakeFiles/hd_exec.dir/run_hausdorff_distance.cu.o"
  "CMakeFiles/hd_exec.dir/run_hausdorff_distance.cu.o.d"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA CXX)
  include(CMakeFiles/hd_exec.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
