# tools/ds4/cmake/ds4_moe.cmake - slice ds4-moe: the routed-expert tier as a library.
#
# Included at the end of the top-level CMakeLists.txt (OPTIONAL, so a tree without this slice still configures).
#
# ONE SOURCE, TWO FLAVOURS.
#   * `ds4_moe_cpu` compiles tools/ds4/ds4_moe.cpp WITHOUT DS4_MOE_CUDA.  That TU contains no CUDA call and
#     references no symbol of strata_kernels / strata_engine / strata_core, so nothing on this target's link line
#     pulls in libcudart.  `test_ds4_moe` links it, which is what makes the gate safe to run on a machine whose
#     GPU is busy (Mal's HARD SAFETY RULE): the binary cannot initialise the driver even by accident.
#   * `ds4_moe_cuda` is the production flavour.  It exists in this slice ONLY to prove the CUDA half COMPILES;
#     nothing in this slice builds a runnable target on top of it.
if(NOT TARGET strata_kernels_cpu OR NOT TARGET ggml-cpu)
  return()
endif()

set(_ds4_moe_include ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4)

add_library(ds4_moe_cpu STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/ds4_moe.cpp)
target_include_directories(ds4_moe_cpu PUBLIC ${_ds4_moe_include})
target_link_libraries(ds4_moe_cpu PUBLIC strata_kernels_cpu strata_artifact ggml ggml-base)

add_executable(test_ds4_moe ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/test_ds4_moe.cpp)
target_link_libraries(test_ds4_moe PRIVATE ds4_moe_cpu)
if(STRATA_BUILD_TESTS)
  # CPU-only by construction (see above); the gate runs it under
  #   nice -n 10 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0
  add_test(NAME test_ds4_moe COMMAND test_ds4_moe)
endif()

if(TARGET CUDA::cudart AND TARGET strata_kernels)
  add_library(ds4_moe_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/ds4_moe.cpp)
  target_compile_definitions(ds4_moe_cuda PRIVATE DS4_MOE_CUDA=1)
  target_include_directories(ds4_moe_cuda PUBLIC ${_ds4_moe_include})
  # `strata_engine` is where `ExpertCache` lives (src/core/expert_cache.cpp; `PinnedArena` is in strata_core, which
  # strata_engine already links) - the same set of libraries `ds4_moe_replay` links.
  target_link_libraries(ds4_moe_cuda PUBLIC strata_engine strata_kernels strata_kernels_cpu strata_artifact ggml
                                           ggml-base CUDA::cudart)
  # run_chunk's MMQ products (Ds4MoeConfig::chunk_mmq) when the build has the prompt MMQ path
  if(TARGET strata_mmq)
    target_compile_definitions(ds4_moe_cuda PRIVATE DS4_MOE_MMQ=1)
    target_link_libraries(ds4_moe_cuda PUBLIC strata_mmq)
  endif()

  # The gate's GPU arm: the SAME source, compiled with DS4_MOE_CUDA and linked against the CUDA flavour.  It is a
  # separate binary on purpose - the CPU gate's safety property (a binary that cannot initialise the driver) is not
  # weakened by it.  `--gpu` adds arm 3 (the four sources against the dequant+F32 reference on 24 real expert
  # slices, ~162 MiB of the GGUF through its mmap); `--timing` is the full-model run and must go through memguard.
  #   nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 build-ds4-cuda/test_ds4_moe_gpu --gpu
  #   bash tools/ds4/memguard.sh 64 62 -- build-ds4-cuda/test_ds4_moe_gpu --timing --routes ... --pred ...
  add_executable(test_ds4_moe_gpu ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/test_ds4_moe.cpp)
  target_compile_definitions(test_ds4_moe_gpu PRIVATE DS4_MOE_CUDA=1)
  target_link_libraries(test_ds4_moe_gpu PRIVATE ds4_moe_cuda)
endif()
