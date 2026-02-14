# DMA-BUF Zero-Copy Test and Benchmark for TFLite VX Delegate
add_executable(dmabuf_benchmark dmabuf_benchmark.cc)

target_include_directories(dmabuf_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/examples
)

# Option to use sysroot's pre-built tensorflow-lite instead of FetchContent version
# This ensures ABI compatibility with target system
option(DMABUF_BENCHMARK_USE_SYSROOT_TFLITE "Use sysroot's tensorflow-lite" OFF)

if(DMABUF_BENCHMARK_USE_SYSROOT_TFLITE AND DEFINED ENV{SDKTARGETSYSROOT})
    message(STATUS "dmabuf_benchmark: Using sysroot tensorflow-lite")
    target_link_libraries(dmabuf_benchmark
        tensorflow-lite
        ${VX_DELEGATE_TARGET}
        vx_custom_op
    )
else()
    target_link_libraries(dmabuf_benchmark
        TensorFlow::tensorflow-lite
        ${VX_DELEGATE_TARGET}
        vx_custom_op
    )
endif()
