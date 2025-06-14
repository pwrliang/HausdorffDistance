include_directories(${OptiX_INCLUDE})
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/thirdparty")

OPTIX_COMPILE_SHADERS("${MODULE_TARGET_DIR}/ptx" PROGRAM_MODULES)

message("PROGRAM_MODULES: ${PROGRAM_MODULES}")
message("ENABLED_ARCHS: ${ENABLED_ARCHS}")
add_executable(hd_exec
        main.cpp
        flags.cc
        run_hausdorff_distance.cu
        autotune_hausdorff_distance.cu
        ${PROGRAM_MODULES}
)

target_link_libraries(hd_exec PRIVATE
        cuda
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${ITK_LIBRARIES}
        ${HdrHistogram_LIBRARIES}
        nlohmann_json::nlohmann_json
        rply
)
target_compile_options(hd_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --fmad=false>)
set_target_properties(hd_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION OFF)
