# tools/ds4/cmake/ds4_dense.cmake - slice ds4-dense: the dense-half decode engine (Ds4Dense) + its gate.
# include()d at the end of the root CMakeLists.txt (OPTIONAL, so a missing file is fine).
#
# build-ds4       : CPU only (the gate runs here)
# build-ds4-cuda  : the identical, backend-agnostic engine sources are compiled and linked against ggml-cuda
#                   as well; the gate itself stays CPU-only and never initialises CUDA.

if(TARGET ggml-cpu)
    add_library(ds4_dense STATIC tools/ds4/ds4_dense.cpp)
    target_include_directories(ds4_dense PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4)
    target_link_libraries(ds4_dense PRIVATE ggml ggml-cpu ggml-base)

    add_executable(test_ds4_dense tools/ds4/test_ds4_dense.cpp)
    target_include_directories(test_ds4_dense PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4)
    target_link_libraries(test_ds4_dense PRIVATE ds4_dense ggml ggml-cpu ggml-base)

    if(TARGET ggml-cuda)
        target_link_libraries(ds4_dense PRIVATE ggml-cuda)
        target_link_libraries(test_ds4_dense PRIVATE ggml-cuda)
    endif()
endif()
