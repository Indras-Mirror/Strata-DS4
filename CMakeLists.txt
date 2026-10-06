# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2.  The kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard
# below refuses anything older: silently building for a pre-Ampere arch would produce a binary that runs
# and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.39 LANGUAGES CXX)
# EXPERIMENTAL: the Intel Arc engine, the SYCL port in sycl/ (PR #423; icpx + oneMKL, docs/INTEL_ARC.md). It is its
# own project that takes the unported sources from src/; with this on, nothing below is configured.
option(STRATA_ENABLE_SYCL "EXPERIMENTAL: build the Intel Arc (SYCL) engine from sycl/ - docs/INTEL_ARC.md" OFF)
if(STRATA_ENABLE_SYCL)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_ENABLE_SYCL is a separate backend; leave STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
  endif()
  message(WARNING "STRATA_ENABLE_SYCL: the Intel Arc engine is experimental (community port, docs/INTEL_ARC.md)")
  add_subdirectory(sycl)
  return()
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
# EXPERIMENTAL, opt-in: an engine for CPUs without AVX2 (#394, #595, #623), compiled on that PC.  ggml-cpu, which then
# computes every CPU expert row, is built for this floor instead of AVX2 or the build host: "avx" (Sandy/Ivy Bridge:
# AVX, no FMA/F16C/AVX2) or "none" (SSE4.2 + POPCNT: Nehalem/Westmere).  Strata's own AVX2/AVX-512 kernels stay in the
# binary behind their CPU checks; the router lookahead gets an AVX1 kernel.  Empty (the default) changes nothing.
set(STRATA_ISA_FLOOR "" CACHE STRING "experimental older-CPU build: empty (default, AVX2 floor), avx, or none (SSE4.2)")
set_property(CACHE STRATA_ISA_FLOOR PROPERTY STRINGS "" avx none)
if(NOT STRATA_ISA_FLOOR STREQUAL "" AND NOT STRATA_ISA_FLOOR STREQUAL "avx" AND NOT STRATA_ISA_FLOOR STREQUAL "none")
  message(FATAL_ERROR "STRATA_ISA_FLOOR must be empty, avx or none (got '${STRATA_ISA_FLOOR}')")
endif()
if(STRATA_ISA_FLOOR AND STRATA_PORTABLE)
  message(FATAL_ERROR "STRATA_ISA_FLOOR (an older-CPU build) and STRATA_PORTABLE (the AVX2 release build) exclude each other")
endif()
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# AMD gfx906 (MI50, wave64): the same .cu sources compiled as HIP through the gfx906 compat layer.  The CUDA runtime and
# cuBLAS names are mapped by the headers in include/strata/platform/hip_compat/, which go first on the include path,
# and CUDA::cudart / CUDA::cublas become aliases of HIP's targets, so every CUDA target below stays as it is.  This is
# separate from STRATA_ENABLE_HIP (the wave32 RDNA backend): with it on, STRATA_ENABLE_CUDA is on and
# STRATA_ENABLE_HIP stays off, so every "CUDA" branch of the tree is the one taken.
option(STRATA_HIP_GFX906 "Build the CUDA targets for AMD gfx906 through HIP (wave64 compat layer)" OFF)
if(STRATA_HIP_GFX906)
  if(STRATA_ENABLE_HIP)
    message(FATAL_ERROR "STRATA_HIP_GFX906 and STRATA_ENABLE_HIP are separate backends; enable only one")
  endif()
  set(STRATA_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
# CUDA: MMQ prompt kernels for Q4_K / Q5_K / Q5_1 (Unsloth's UD-Q4_K_XL experts).  Off by default: the engine loads
# every kernel when it starts (CUDA_MODULE_LOADING=EAGER), so these took ~5 MB of VRAM - 1-3 expert slots - from every
# model, and the UD-Q4_K_XL prompt is SSD-bound with or without them (docs/UNSLOTH_Q4.md)
option(STRATA_MMQ_KQUANTS "Build the CUDA MMQ prompt kernels for Q4_K / Q5_K / Q5_1 experts" OFF)
option(STRATA_ORCA_Q4KS_MMQ "Build the GGML Q5_0 MMQ instance for OrcaRouter Q4_K_S" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

if(STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  add_executable(conversation_memory_test src/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  add_test(NAME conversation_memory_test COMMAND conversation_memory_test)
endif()

if(STRATA_HIP_GFX906)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures" FORCE)
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  target_link_libraries(CUDA::cudart INTERFACE hip::host)
  target_compile_definitions(CUDA::cudart INTERFACE STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(CUDA::cudart BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_library(CUDA::cublas INTERFACE IMPORTED)
  target_link_libraries(CUDA::cublas INTERFACE roc::hipblas CUDA::cudart)
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_compile_definitions(STRATA_HIP_GFX906=1)
  message(STATUS "Strata: HIP (gfx906 compat) enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_CUDA)
  # #585: CMake links CUDA code with the STATIC runtime by default, and the targets below also link CUDA::cudart (the
  # DLL's import library).  MSVC with a CUDA 12.x toolkit - the experimental Pascal/Volta build - then fails with
  # LNK1169 (every cudart symbol defined twice); GNU ld lets it pass.  That build links the shared runtime alone.  The
  # ready-made engine (CUDA 13, no STRATA_EXPERIMENTAL_SM60) is unchanged; -DCMAKE_CUDA_RUNTIME_LIBRARY overrides.
  if(MSVC AND STRATA_EXPERIMENTAL_SM60 AND NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  option(STRATA_EXPERIMENTAL_SM60 "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Turing port: developed and measured on sm_120 (RTX 50); the tf32 mma in the QSA scorer has a
    # portable fp32-FMA fallback, so RTX 20 (75), 30 (86), RTX 40 (89) and RTX 50 (120) build too.
    # Pre-sm_75 is refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    # STRATA_EXPERIMENTAL_SM60: a Pascal (sm_60, GP100-class) or Volta (sm_70, V100 / Titan V, #236) build.
    # These sit below every architecture the kernels were written for, so two intrinsics are replaced below
    # their arch - `__dp4a` (6.1+) and `__nanosleep` (7.0+) - in include/strata/kernels/dp4a.hpp, which is
    # bit-exact for the first and merely a shorter pause for the second (sm_70 has both).  Community-tested,
    # not in the ready-made engine; RTX 20/30/40/50 cards must use the release engine instead.  It needs a CUDA
    # 12.x toolkit: CUDA 13 dropped sm_60 and sm_70 (nvcc fails CMake's compiler test with "compute_70").
    if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.  (A Pascal sm_60 or Volta sm_70 build is "
        "-DSTRATA_EXPERIMENTAL_SM60=ON; upstream does not support it.)")
    endif()
    if(_base LESS 60)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES '${CMAKE_CUDA_ARCHITECTURES}': even the experimental build "
                          "needs compute capability 6.0 (sm_60) or newer")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  if(STRATA_EXPERIMENTAL_SM60)
    # every target sees it: the Pascal / Volta code paths (the Volta prompt attention, PR #600) are compiled into this
    # build only, so the ready-made engine (sm_75+) keeps exactly the code it had
    add_compile_definitions(STRATA_EXPERIMENTAL_SM60=1)
  endif()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target ${STRATA_HIP_BLAS_TARGETS})
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)
  # DSV4 geometry + full tensor map (header mmap only; the real-file part skips without $DS4_GGUF/argv[1])
  add_executable(ds4_geometry_test tests/core/ds4_geometry_test.cpp)
  target_link_libraries(ds4_geometry_test PRIVATE strata_artifact)
  add_test(NAME ds4_geometry_test COMMAND ds4_geometry_test $ENV{DS4_GGUF})
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(STRATA_ENABLE_CUDA AND STRATA_EXPERIMENTAL_SM60)
    # the runtime device check (src/core/device.cu) admits the cards this build was made for (#236)
    target_compile_definitions(strata_core PRIVATE STRATA_EXPERIMENTAL_SM60=1)
  endif()
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON.  CUDA only: HIP compiles the tensor-core kernel out.
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_CUDA)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels CUDA::cudart)
  endif()
  # #512: the top-k's active-block bound (Turing's dispatch, forced on any card by STRATA_TOPK_ACTIVE_ANY) against
  # the capacity rule, selected ids bitwise
  if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
    add_executable(qsa_topk_active_parity src/kernels/qsa_topk_active_parity.cpp)
    target_include_directories(qsa_topk_active_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
    set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
  # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
  # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
  if((STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN) AND STRATA_ENABLE_HIP)
    add_executable(qsa_select_bench src/kernels/qsa_select_bench.cpp)
    set_source_files_properties(src/kernels/qsa_select_bench.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(qsa_select_bench PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME qsa_select_bench COMMAND qsa_select_bench 32768 256 3 262144)
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "$<IF:$<BOOL:${STRATA_HIP_GFX906}>,,--use_fast_math>")
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  # This oracle is a CUDA ggml build; do not pull CUDA libraries into a HIP build.
  if(STRATA_ENABLE_CUDA AND STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/remote_expert_opt.cu
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
  endif()
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/expert_profile_save_test.cpp")
    enable_testing()
    add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
    target_link_libraries(expert_profile_save_test PRIVATE strata_engine)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  endif()
  if(STRATA_BUILD_TESTS AND STRATA_NATIVE_EXPERTS)
    # native_experts.txt v2-v4 (the real packs' copies and the per-role shard column) and the arena filled from
    # synthetic GGUF shards at the real expert dimensions
    add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
    target_link_libraries(expert_layout_test PRIVATE strata_engine)
    add_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
    set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
    # #326: a native pack's BF16 (--compat-bf16) PLE key stays in the arena; a quantized key row loads as before
    add_executable(native_dense_ple_key_test tests/core/native_dense_ple_key_test.cpp)
    target_link_libraries(native_dense_ple_key_test PRIVATE strata_engine)
    add_test(NAME native_dense_ple_key_test COMMAND native_dense_ple_key_test)
    set_tests_properties(native_dense_ple_key_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    # #369: per-layer admission on a sized (native pack) cache starts each layer in its own range
    add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
    target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
    add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
    set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
    target_link_libraries(expert_cache_segmented_test PRIVATE strata_engine)
    add_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
    set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  if(STRATA_BUILD_CONVERSATION_TESTS)
    add_executable(conversation_snapshot_test src/core/conversation_snapshot_test.cpp)
    target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
    add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_executable(conversation_validation_test src/core/conversation_validation_test.cpp)
    target_link_libraries(conversation_validation_test PRIVATE strata_engine)
    add_test(NAME conversation_validation_test COMMAND conversation_validation_test)
    set_tests_properties(conversation_validation_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    if(NOT STRATA_ENABLE_HIP AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_executable(conversation_transfer_test src/core/conversation_validation_test.cpp)
      target_compile_definitions(conversation_transfer_test PRIVATE CONVERSATION_TEST_TRANSFERS)
      target_link_options(conversation_transfer_test PRIVATE
        -Wl,--wrap=cudaMemcpy -Wl,--wrap=cudaDeviceSynchronize -Wl,--wrap=cudaGetLastError)
      target_link_libraries(conversation_transfer_test PRIVATE strata_engine)
      add_test(NAME conversation_transfer_test COMMAND conversation_transfer_test)
      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
    endif()
  endif()

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
    target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)

    add_executable(tune_hipblaslt tools/hip/tune_hipblaslt.cpp)
    target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
  endif()

  add_executable(strata src/program/generate.cpp)
  target_include_directories(strata PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    add_executable(hip_prefill_gemm tests/hip/prefill_gemm.cpp)
    target_link_libraries(hip_prefill_gemm PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_gemm COMMAND hip_prefill_gemm)
    list(APPEND _strata_hip_tests hip_prefill_gemm)
    add_executable(hip_q2_zero tests/hip/q2_zero.cpp)
    target_link_libraries(hip_q2_zero PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_q2_zero COMMAND hip_q2_zero)
    list(APPEND _strata_hip_tests hip_q2_zero)
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/mapped_alias.cpp")
      # #325: reports what a hipHostGetDevicePointer alias is on this stack (on Windows: the host pointer, and a
      # D2D copy into it does not land - hip_handoff's timeout there). A diagnosis: exits 0 whenever it ran.
      add_executable(hip_mapped_alias tests/hip/mapped_alias.cpp)
      target_link_libraries(hip_mapped_alias PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_mapped_alias COMMAND hip_mapped_alias)
      set_tests_properties(hip_mapped_alias PROPERTIES SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_mapped_alias)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    # S6: the AMD fast router against the portable kernel, bitwise
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_link_libraries(hip_router_fast PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_router_fast COMMAND hip_router_fast)
    list(APPEND _strata_hip_tests hip_router_fast)
    # S6: the RDNA4 matrix-core prompt attention (STRATA_HIP_WMMA) against the FP32 kernel and FP64; skips off gfx12
    add_executable(hip_prompt_attn_wmma src/kernels/qsa_prompt_attn_parity.cpp)
    target_link_libraries(hip_prompt_attn_wmma PRIVATE strata_kernels strata_hip_runtime)
    add_test(NAME hip_prompt_attn_wmma COMMAND hip_prompt_attn_wmma 32768 2048 3)
    set_tests_properties(hip_prompt_attn_wmma PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    list(APPEND _strata_hip_tests hip_prompt_attn_wmma)
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  # TODO 24: the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
  # and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
  # BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
  # explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  # the generator needs an interpreter WITH numpy: probe the configured one, the setup's .venv, and the
  # system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                                "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
                                /usr/bin/python3
                                python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures
             COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    # exit 3: numpy or the vendored gguf-py is missing (a trimmed third_party) - a skip, not a failure
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA tests: no python3 with numpy was found (the configured interpreter and .venv "
                   "were probed) - the iq_parity fixtures and its test are not registered; the fixtures "
                   "need python3 with numpy and the repository's vendored gguf-py")
  endif()
  if(STRATA_ENABLE_CUDA)
    # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
    # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
    add_executable(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)
    target_include_directories(mmvq_multi_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
    if(STRATA_HIP_GFX906)   # the CUDA layout, where the negative control has power (the wave64 one is exact at every T)
      add_test(NAME mmvq_multi_parity_cuda_layout COMMAND mmvq_multi_parity)
      set_tests_properties(mmvq_multi_parity_cuda_layout PROPERTIES ENVIRONMENT "STRATA_MMVQ_WAVE=0")
    endif()
  endif()
  # the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (GPU, synthetic, no model; --bench adds old/new timings)
  add_executable(iq_multi_parity src/kernels/iq_multi_parity.cpp)
  target_include_directories(iq_multi_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  # the prompt path's DeltaNet recurrence: gdn_rec_kh_kernel against gdn_rec_cols_pipe_kernel, bitwise (GPU,
  # synthetic, no model; --bench adds timings).  NVIDIA only: cp.async and the CUDA occupancy API (not on gfx906)
  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
    add_executable(gdn_rec_parity src/prefill/gdn_rec_parity.cu)
    target_link_libraries(gdn_rec_parity PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME gdn_rec_parity COMMAND gdn_rec_parity)
    # the prompt path's BF16 products against cuBLAS's own BF16 GEMM, within fp32 rounding (sm_7x runs them on the
    # FP16 tensor cores; GPU, synthetic, no model; --bench adds timings)
    add_executable(gemm_bf16_parity src/prefill/gemm_bf16_parity.cu)
    target_link_libraries(gemm_bf16_parity PRIVATE strata_prefill )
    add_test(NAME gemm_bf16_parity COMMAND gemm_bf16_parity)
  endif()
  # native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the launches before them (GPU, synthetic,
  # no model; --bench adds v1/new timings of a window's 48 calls in a graph)
  add_executable(native_grouped_parity src/kernels/native_grouped_parity.cpp)
  target_include_directories(native_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(fused_gr_bench src/kernels/fused_gr_bench.cpp)
  target_include_directories(fused_gr_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(fused_gr_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=0")
  # the same fixtures on the other two sampled paths: the one-block kernel and the 0.1.20 kernel
  add_test(NAME sampler_parity_one_block COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_one_block PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=0;STRATA_SAMPLER_ONE_BLOCK=1")
  add_test(NAME sampler_parity_old COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_old PROPERTIES ENVIRONMENT "STRATA_OLD_SAMPLER=1;STRATA_SAMPLER_ONE_BLOCK=0")
  # S19: the decode QSA top-k and the greedy argmax on thread-block clusters (sm_90+) vs the one-CTA kernels, bitwise,
  # captured in a graph too (GPU, synthetic, no model; skips below sm_90); --bench: per-call times
  if(STRATA_ENABLE_CUDA)
    add_executable(decode_cluster_parity src/kernels/decode_cluster_parity.cpp)
    target_include_directories(decode_cluster_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(decode_cluster_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME decode_cluster_parity COMMAND decode_cluster_parity --selftest)
  endif()

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  # the verify window's routing in one multi-column BF16 projection + one top-k, bitwise the per-token calls
  add_executable(route_window_parity src/kernels/route_window_parity.cpp)
  target_include_directories(route_window_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(route_window_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME route_window_parity COMMAND route_window_parity)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)
  if(STRATA_HIP_GFX906)
    # the S6 router (STRATA_HIP_ROUTER_FAST=1 on gfx906) against the portable kernel, bitwise, on wave64
    add_executable(hip_router_fast tests/hip/router_fast.cpp)
    target_include_directories(hip_router_fast PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(hip_router_fast PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # the QSA top-k the engine dispatches against the original kernel: identical ids on both sides of the register
  # kernel's reach, with tied and with equal scores (synthetic, no model)
  add_executable(qsa_topk_parity src/kernels/qsa_topk_parity.cpp)
  target_include_directories(qsa_topk_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_topk_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_topk_parity COMMAND qsa_topk_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # PR #120: the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
  add_executable(kv_hybrid_parity src/kernels/kv_hybrid_parity.cpp)
  target_include_directories(kv_hybrid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # The rewritten per-hit and grouped expert kernels vs the previous ones, bitwise (GPU, synthetic, no model).
  # `--bench` adds previous/new timings on engine-like shapes.
  add_executable(s2_expert_grouped_parity src/kernels/s2_expert_grouped_parity.cpp)
  target_include_directories(s2_expert_grouped_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_expert_grouped_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/native_mmvq_multi.cpp")
    add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
    target_include_directories(native_mmvq_multi PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_mmvq_multi PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pinned_shared_test src/core/pinned_shared_test.cpp)
    target_link_libraries(pinned_shared_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME pinned_shared_test COMMAND pinned_shared_test)
  endif()

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test runs ON THE TARGET PC. `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  if(STRATA_ENABLE_CUDA)
    add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)
  else()
    add_test(NAME hip_device_selftest COMMAND strata-device --selftest)
  endif()

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu")
    add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
    target_include_directories(graph_registry_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(graph_registry_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME graph_registry_test COMMAND graph_registry_test)
  endif()

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/pinned_capture.cu")
    add_executable(pinned_capture bench/micro/pinned_capture.cu)
    target_include_directories(pinned_capture PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pinned_capture PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  if(STRATA_ENABLE_CUDA)
    set(STRATA_CUDA_TESTS ON)
  endif()
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()
if(STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  # coupled draft sampling's counter and penalty-ring arithmetic (header-only, CPU)
  add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
  target_include_directories(coupled_draft_test PRIVATE include)
  add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_ORCA_Q4KS_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORCA_Q4KS_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  # ggml's own CUDA backend is off unless asked for: the DS4 dense half (tools/ds4/ds4_dense) runs its graphs on a
  # ggml backend and needs ggml-cuda to run them on the GPU (-DSTRATA_GGML_CUDA=ON, build-ds4-gpu).
  option(STRATA_GGML_CUDA "Build ggml's CUDA backend too (for the DS4 dense half on the GPU)" OFF)
  if(STRATA_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
  else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  endif()
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                                           src/kernels/cpu/kq_avx1.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    # the router lookahead's AVX1 dot (older CPUs; called behind cpu_avx1_ok): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
  endif()
  if(STRATA_ISA_FLOOR STREQUAL "avx")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_AVX=1)
  elseif(STRATA_ISA_FLOOR STREQUAL "none")
    target_compile_definitions(strata_kernels_cpu PRIVATE STRATA_ISA_FLOOR_NONE=1)
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  # the router lookahead's dot on each rung this CPU has (AVX2, the older CPUs' AVX1) against a double reference
  add_executable(router_dot_parity src/kernels/cpu/router_dot_parity.cpp)
  target_link_libraries(router_dot_parity PRIVATE strata_kernels_cpu)
  if(STRATA_BUILD_TESTS)
    add_test(NAME router_dot_parity COMMAND router_dot_parity)
    add_test(NAME router_dot_parity_avx COMMAND ${CMAKE_COMMAND} -E env STRATA_FORCE_ISA=avx $<TARGET_FILE:router_dot_parity>)
  endif()
  set(_strata_mmq_types q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
  # #296: OrcaRouter Q4_K_S's Q5_0 experts (Q4_K and Q5_1 come with STRATA_MMQ_KQUANTS on CUDA, S5's branch)
  if(STRATA_ORCA_Q4KS_MMQ)
    list(APPEND _strata_mmq_types q5_0)
  endif()
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    # #136: the Q2_0 pack's prompt experts on Strata's own int8 kernels (moe_fused.cu, opt-in STRATA_PF_FUSED=1; CUDA
    # only: the HIP branch below keeps MMQ, and an sm_75 device or an sm_75-only build reports them unavailable)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu
                src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
    # q8_0: the drafter's matrices (E-9); STRATA_ORCA_Q4KS_MMQ adds q5_0 (#296, in _strata_mmq_types), and
    # STRATA_MMQ_KQUANTS q4_k q5_k q5_1, the experts of Unsloth's UD-Q4_K_XL (docs/UNSLOTH_Q4.md; the dispatch
    # follows eddoursul/Strata 7706c1b)
    set(_strata_mmq_cuda_types ${_strata_mmq_types})
    if(STRATA_MMQ_KQUANTS)
      list(APPEND _strata_mmq_cuda_types q4_k q5_k q5_1)
    endif()
    foreach(_t IN LISTS _strata_mmq_cuda_types)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    if(STRATA_BUILD_TESTS)
      # #136: the fused int8 experts against a double-precision reference and the MMQ path (exit 77 below sm_80)
      add_executable(prefill_fused_moe_test tests/cuda/prefill_fused_moe_test.cpp)
      target_link_libraries(prefill_fused_moe_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_moe_test COMMAND prefill_fused_moe_test)
      set_tests_properties(prefill_fused_moe_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
      # #136: the same for the native packs' formats (IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL)
      add_executable(prefill_fused_iq_test tests/cuda/prefill_fused_iq_test.cpp)
      target_link_libraries(prefill_fused_iq_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_fused_iq_test COMMAND prefill_fused_iq_test)
      set_tests_properties(prefill_fused_iq_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_MMQ_KQUANTS)
      # the UD-Q4_K_XL expert formats through MMQ against ggml's dequantized weights (synthetic, a few seconds)
      add_executable(prefill_mmq_kquant_test tests/cuda/prefill_mmq_kquant_test.cpp)
      target_link_libraries(prefill_mmq_kquant_test PRIVATE strata_mmq ggml-base ${_strata_gpu_runtime_target})
      add_test(NAME prefill_mmq_kquant_test COMMAND prefill_mmq_kquant_test)
      set_tests_properties(prefill_mmq_kquant_test PROPERTIES TIMEOUT 120)
    endif()
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    if(STRATA_GGML_DIR)
      set(_strata_gg "${STRATA_GGML_DIR}/ggml")
    else()
      set(_strata_gg "${strata_llamacpp_SOURCE_DIR}/ggml")
    endif()
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_mmq}/quantize.cu")
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
    if(NOT STRATA_HIP_GFX906)
      # DS4 phase 2 gate: one real DeepSeek-V4 expert (gate/up IQ2_XXS, down Q2_K) through the CPU pool and the
      # GPU native MMVQ path, against ggml's dequantizer + an F32 matmul.  The 0731 artifact comes from
      # argv[1]/$DS4_GGUF; without it the binary prints "skipped" and exits 0.  --cpu-only runs no CUDA.
      add_executable(ds4_expert_parity src/kernels/ds4_expert_parity.cpp)
      target_include_directories(ds4_expert_parity PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(ds4_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
      if(STRATA_BUILD_TESTS)
        add_test(NAME ds4_expert_parity COMMAND ds4_expert_parity "$ENV{DS4_GGUF}")
      endif()
      # DS4 phase 4a: the DeepSeek-V4 expert engine replayed on recorded routes (tools/ds4/moe_replay.cpp).
      # It links the real machinery - strata_engine's ExpertCache, the CPU ExpertPool, the native grouped
      # kernels, PinnedArena - and drives them from ds4routes.bin.  --correctness --cpu-only needs no GPU.
      add_executable(ds4_moe_replay tools/ds4/moe_replay.cpp tools/ds4/moe_gap.cu)
      target_include_directories(ds4_moe_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${_strata_gpu_include_directories})
      target_link_libraries(ds4_moe_replay PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact
        ${_strata_gpu_runtime_target})
    endif()
    if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
      # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
      # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
      # q4_K/q5_0 and q5_0/q8_0: a plain llama-quantize "_M" tier GGUF (not Unsloth's dynamic method) commonly
      # puts Q5_0 on down-projections (sometimes gate/up too) where this model's native_experts.txt had no GPU
      # kernel for it before; q5_0/q5_0 (both roles Q5_0) is left out of this list since stacking the same
      # lower-precision format on both roles pushes the GPU path over this test's 3% threshold (quantization
      # noise, not a kernel bug -- see the PR).
      foreach(_pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0)
        string(REPLACE "/" "_" _name "${_pair}")
        add_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
        set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
      endforeach()
      add_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
      # #290: the BF16 token embedding (--embd-gguf) gathered and dequantized bit for bit
      add_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
      # a pair without GPU kernels (Q6_K gate/up) is refused by native_expert_supported, not run
      add_test(NAME native_expert_parity_refuses_q6_K COMMAND native_expert_parity --synthetic q6_K/q8_0)
      set_tests_properties(native_expert_parity_refuses_q6_K PROPERTIES WILL_FAIL TRUE)
    endif()
  endif()
  # #296's real-weight checks (OrcaRouter Q4_K_S files; docs/ORCA_Q4_K_S.md) - test programs, not part of a release
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
    add_executable(ple_q5_parity src/kernels/ple_q5_parity.cpp)
    target_link_libraries(ple_q5_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    add_executable(q5_projection_parity src/kernels/q5_projection_parity.cpp)
    target_include_directories(q5_projection_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(q5_projection_parity PRIVATE strata_kernels strata_artifact ggml-base ${_strata_gpu_runtime_target})
    # #291: the FP8 PLE table (tools/ple_fp8_pack.py) through PleTable vs the checkpoint's own values
    add_executable(ple_fp8_parity src/kernels/ple_fp8_parity.cpp)
    target_link_libraries(ple_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/hit_cpu_order_parity.cu")
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
endif()

if(STRATA_BUILD_TESTS)
  if(WIN32)
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pool_affinity_test src/kernels/cpu/pool_affinity_linux_test.cpp)
    target_link_libraries(pool_affinity_test PRIVATE strata_kernels_cpu)
    add_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  endif()
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)   # #93: it needs the device dequantizers and a GPU runtime
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  if(UNIX AND NOT APPLE)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp src/platform/direct_file.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
    add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

# ---- HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) is compiled as HIP ----
if(STRATA_HIP_GFX906)
  file(GLOB_RECURSE _strata_cu CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/tools/vision/*.cu)
  if(_strata_mmq)
    file(GLOB _strata_mmq_cu ${_strata_mmq}/*.cu ${_strata_mmq}/template-instances/*.cu)
    list(APPEND _strata_cu ${_strata_mmq_cu})
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP)
  endif()
  set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)
endif()

# ---- ds4 reference forward pass (Phase 3) ---------------------------------
# tools/ds4 is not part of the engine, but ds4_ref is the Phase-3 correctness oracle: a plain CPU ggml
# graph that emits golden_dump-format activations for tools/ds4/compare_golden.py.  CPU-only, no CUDA.
if(TARGET ggml-cpu)
  add_executable(ds4_ref tools/ds4/ds4_ref.cpp)
  target_include_directories(ds4_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_ref PRIVATE ggml ggml-cpu ggml-base)
endif()

# ---- ds4 decode engine (Phase 4b proper) ---------------------------------
# Each engine slice owns exactly one of these files (conductor plan 2026-10-06); OPTIONAL so a missing one is fine.
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_dense.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_moe.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_engine.cmake OPTIONAL)
