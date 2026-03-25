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
#include <stdbool.h>

struct TfLiteDelegate;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DMA-BUF tensor information returned by hal_dmabuf_get_tensor_info.
 *
 * For the VX delegate, each tensor has its own DMA-BUF so offset is always 0.
 * Other backends (e.g. Neutron) may pack multiple tensors into a single buffer
 * with non-zero offsets.
 */
typedef struct {
  int fd;          /**< DMA-BUF file descriptor */
  size_t offset;   /**< Byte offset within the DMA-BUF */
  size_t size;     /**< Tensor size in bytes */
} hal_dmabuf_tensor_info;

/**
 * Check if DMA-BUF zero-copy is supported by this delegate.
 *
 * @param delegate The TFLite delegate instance
 * @return true if DMA-BUF is supported, false otherwise
 */
bool hal_dmabuf_is_supported(struct TfLiteDelegate* delegate);

/**
 * Get the current delegate instance (singleton).
 *
 * @return The delegate instance, or NULL if none has been created
 */
struct TfLiteDelegate* hal_dmabuf_get_instance(void);

/**
 * Get DMA-BUF information for a tensor.
 *
 * Populates info with the file descriptor, byte offset, and size for the
 * DMA-BUF backing the given tensor index.
 *
 * @param delegate   The TFLite delegate instance
 * @param tensor_index  TFLite tensor index
 * @param info       Output struct to populate
 * @param info_size  sizeof(*info) for forward compatibility
 * @return 0 on success, -1 on failure
 */
int hal_dmabuf_get_tensor_info(struct TfLiteDelegate* delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info* info,
                               size_t info_size);

/**
 * Synchronize a tensor's DMA-BUF for device (NPU) access.
 *
 * Call after CPU writes to flush caches before NPU reads.
 *
 * @param delegate      The TFLite delegate instance
 * @param tensor_index  TFLite tensor index
 * @return 0 on success, -1 on failure
 */
int hal_dmabuf_sync_for_device(struct TfLiteDelegate* delegate,
                               int tensor_index);

/**
 * Synchronize a tensor's DMA-BUF for CPU access.
 *
 * Call before CPU reads to invalidate caches after NPU writes.
 *
 * @param delegate      The TFLite delegate instance
 * @param tensor_index  TFLite tensor index
 * @return 0 on success, -1 on failure
 */
int hal_dmabuf_sync_for_cpu(struct TfLiteDelegate* delegate,
                            int tensor_index);

#ifdef __cplusplus
}
#endif

#endif  // HAL_DMABUF_H_
