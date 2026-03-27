/****************************************************************************
 *
 *    Copyright (c) 2025 Au-Zone Technologies
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef HAL_DMABUF_H_
#define HAL_DMABUF_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque delegate handle. */
typedef void *hal_delegate_t;

/** Element data types. */
typedef enum hal_dtype {
    HAL_DTYPE_U8  = 0,
    HAL_DTYPE_I8  = 1,
    HAL_DTYPE_U16 = 2,
    HAL_DTYPE_I16 = 3,
    HAL_DTYPE_U32 = 4,
    HAL_DTYPE_I32 = 5,
    HAL_DTYPE_U64 = 6,
    HAL_DTYPE_I64 = 7,
    HAL_DTYPE_F16 = 8,
    HAL_DTYPE_F32 = 9,
    HAL_DTYPE_F64 = 10
} hal_dtype;

#define HAL_DMABUF_MAX_NDIM 8

/**
 * DMA-BUF tensor information returned by hal_dmabuf_get_tensor_info.
 *
 * For the VX delegate, each tensor has its own DMA-BUF so offset is always 0.
 * Other backends (e.g. Neutron) may pack multiple tensors into a single buffer
 * with non-zero offsets.
 */
typedef struct hal_dmabuf_tensor_info {
    size_t size;                          /**< Buffer size in bytes */
    size_t offset;                        /**< Byte offset within the DMA-BUF */
    size_t shape[HAL_DMABUF_MAX_NDIM];    /**< Tensor dimensions */
    size_t ndim;                          /**< Number of valid entries in shape */
    int fd;                               /**< DMA-BUF fd (borrowed, do NOT close) */
    hal_dtype dtype;                      /**< Element data type */
} hal_dmabuf_tensor_info;

/** Camera adaptor format information. */
typedef struct hal_camera_adaptor_format_info {
    int input_channels;     /**< e.g., 4 for RGBA */
    int output_channels;    /**< e.g., 3 for RGB */
    char fourcc[8];         /**< V4L2 FourCC string, NUL-terminated */
} hal_camera_adaptor_format_info;

/**
 * Get the current delegate instance (singleton).
 *
 * @return The delegate instance, or NULL if none has been created
 */
hal_delegate_t hal_dmabuf_get_instance(void);

/**
 * Check if DMA-BUF zero-copy is supported by this delegate.
 *
 * @param delegate The delegate instance
 * @return 1 if DMA-BUF is supported, 0 otherwise
 */
int hal_dmabuf_is_supported(hal_delegate_t delegate);

/**
 * Get DMA-BUF information for a tensor.
 *
 * Populates info with the file descriptor, byte offset, size, shape, and
 * data type for the DMA-BUF backing the given tensor index.
 *
 * @param delegate      The delegate instance
 * @param tensor_index  TFLite tensor index
 * @param info          Output struct to populate
 * @param info_size     sizeof(*info) for forward compatibility
 * @return 0 on success, -1 on failure (errno set)
 */
int hal_dmabuf_get_tensor_info(hal_delegate_t delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info *info,
                               size_t info_size);

/**
 * Synchronize a tensor's DMA-BUF for device (NPU) access.
 *
 * Call after CPU writes to flush caches before NPU reads.
 *
 * @param delegate      The delegate instance
 * @param tensor_index  TFLite tensor index
 * @return 0 on success, -1 on failure (errno set)
 */
int hal_dmabuf_sync_for_device(hal_delegate_t delegate, int tensor_index);

/**
 * Synchronize a tensor's DMA-BUF for CPU access.
 *
 * Call before CPU reads to invalidate caches after NPU writes.
 *
 * @param delegate      The delegate instance
 * @param tensor_index  TFLite tensor index
 * @return 0 on success, -1 on failure (errno set)
 */
int hal_dmabuf_sync_for_cpu(hal_delegate_t delegate, int tensor_index);

/**
 * Check if a camera adaptor format is supported.
 *
 * @param delegate  The delegate instance (unused, reserved)
 * @param format    Camera adaptor format string (e.g. "RGBA_TO_RGB")
 * @return 1 if supported, 0 otherwise
 */
int hal_camera_adaptor_is_supported(hal_delegate_t delegate,
                                    const char *format);

/**
 * Get format information for a camera adaptor.
 *
 * @param delegate   The delegate instance (unused, reserved)
 * @param format     Camera adaptor format string
 * @param info       Output struct to populate
 * @param info_size  sizeof(*info) for forward compatibility
 * @return 0 on success, -1 on failure (errno set)
 */
int hal_camera_adaptor_get_format_info(hal_delegate_t delegate,
                                       const char *format,
                                       hal_camera_adaptor_format_info *info,
                                       size_t info_size);

#ifdef __cplusplus
}
#endif

#endif  /* HAL_DMABUF_H_ */
