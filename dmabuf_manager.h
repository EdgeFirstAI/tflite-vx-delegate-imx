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

#ifndef TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_MANAGER_H_
#define TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_MANAGER_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tim/vx/tensor.h"
#include "tim/vx/graph.h"
#include "vx_delegate_dmabuf.h"

namespace vx {
namespace delegate {

/**
 * Holds a DRM GEM handle that keeps a persistent dma_buf_attach alive.
 *
 * On cached CMA heaps (linux,cma), the kernel's begin_cpu_access iterates
 * over buffer->attachments to perform cache maintenance. Without any active
 * attachments, DMA_BUF_IOCTL_SYNC is a no-op — no cache invalidation or
 * flush occurs.
 *
 * By importing the DMA-buf fd through the GPU DRM driver
 * (DRM_IOCTL_PRIME_FD_TO_HANDLE), the driver creates a persistent
 * dma_buf_attach that makes DMA_BUF_IOCTL_SYNC effective.
 *
 * The GEM handle and DRM fd must remain open for the buffer's lifetime.
 * Closing either removes the attachment and makes sync a no-op again.
 */
class DrmAttachment {
 public:
  /**
   * Create a DRM attachment for a DMA-BUF fd.
   * Returns nullptr if /dev/dri/renderD128 is not available.
   */
  static std::unique_ptr<DrmAttachment> Create(int dmabuf_fd);

  ~DrmAttachment();

  DrmAttachment(const DrmAttachment&) = delete;
  DrmAttachment& operator=(const DrmAttachment&) = delete;

 private:
  DrmAttachment(int drm_fd, uint32_t gem_handle);

  int drm_fd_;
  uint32_t gem_handle_;
};

/**
 * Entry tracking a registered or allocated DMA-BUF.
 */
struct DmaBufEntry {
  int fd;                          // DMA-BUF file descriptor
  size_t size;                     // Buffer size in bytes
  VxDmaBufOwnership ownership;     // Who owns the buffer
  VxDmaBufSyncMode default_sync;   // Default sync mode
  int tensor_index;                // Bound tensor index (-1 if not bound)
  std::shared_ptr<tim::vx::Tensor> tim_tensor;  // TIM-VX tensor (created lazily)
  bool is_input;                   // True if bound to input tensor
  bool is_output;                  // True if bound to output tensor
  std::unique_ptr<DrmAttachment> drm_attachment;  // Keeps DMA_BUF_IOCTL_SYNC effective
};

/**
 * Manages DMA-BUF registrations, allocations, and tensor bindings.
 *
 * This class handles:
 * - Registration of externally-allocated dmabufs (import mode)
 * - Allocation of dmabufs from dma_heap (export mode)
 * - Cache synchronization for CPU/NPU coherency
 * - Binding dmabufs to TIM-VX tensors for zero-copy inference
 */
class DmaBufManager {
 public:
  DmaBufManager();
  ~DmaBufManager();

  /**
   * Initialize the manager. Opens the dma_heap device.
   * @param heap_path Path to dma_heap device (default: /dev/dma_heap/linux,cma)
   * @return true if initialization succeeded
   */
  bool Initialize(const char* heap_path = nullptr);

  /**
   * Check if DMA-BUF is supported on this platform.
   */
  bool IsSupported() const { return dma_heap_fd_ >= 0; }

  // === Import Mode (client-provided buffers) ===

  /**
   * Register an externally-allocated DMA-BUF.
   * @param fd DMA-BUF fd (caller retains ownership)
   * @param size Buffer size
   * @param sync_mode Default sync mode
   * @return Handle for future operations, or kTfLiteNullBufferHandle on failure
   */
  TfLiteBufferHandle RegisterBuffer(int fd, size_t size,
                                     VxDmaBufSyncMode sync_mode);

  /**
   * Unregister a buffer (does not close fd for client-owned buffers).
   */
  TfLiteStatus UnregisterBuffer(TfLiteBufferHandle handle);

  // === Export Mode (delegate-allocated buffers) ===

  /**
   * Allocate a DMA-BUF from the heap.
   * @param size Required size
   * @param alignment Required alignment (0 for default)
   * @param ownership Who owns the buffer
   * @param desc Output descriptor with fd
   * @return Handle, or kTfLiteNullBufferHandle on failure
   */
  TfLiteBufferHandle AllocateBuffer(size_t size, size_t alignment,
                                     VxDmaBufOwnership ownership,
                                     VxDmaBufDesc* desc);

  /**
   * Release a delegate-allocated buffer (closes fd).
   */
  TfLiteStatus ReleaseBuffer(TfLiteBufferHandle handle);

  // === Tensor Binding ===

  /**
   * Bind a buffer handle to a tensor index.
   * The TIM-VX tensor will be created lazily on first inference.
   */
  TfLiteStatus BindToTensor(TfLiteBufferHandle handle, int tensor_index,
                            bool is_input, bool is_output);

  /**
   * Create or get the TIM-VX tensor for a bound dmabuf.
   * This creates the tensor with DmaBufferDesc if not already created.
   */
  std::shared_ptr<tim::vx::Tensor> GetOrCreateTensor(
      TfLiteBufferHandle handle,
      std::shared_ptr<tim::vx::Graph>& graph,
      const tim::vx::TensorSpec& spec);

  /**
   * Update the TIM-VX tensor reference after layout inference.
   */
  void UpdateTensorAfterLayoutInference(
      TfLiteBufferHandle handle,
      std::shared_ptr<tim::vx::Tensor> inferred_tensor);

  // === Synchronization ===

  /**
   * Begin CPU access (invalidate cache for reads).
   */
  TfLiteStatus BeginCpuAccess(TfLiteBufferHandle handle, VxDmaBufSyncMode mode);

  /**
   * End CPU access (flush cache for writes).
   */
  TfLiteStatus EndCpuAccess(TfLiteBufferHandle handle, VxDmaBufSyncMode mode);

  // === Buffer Cycling ===

  /**
   * Set the active buffer handle for a tensor (buffer cycling).
   * This determines which buffer from a pool will be used on next inference.
   * The handle must have been previously bound to this tensor via BindToTensor().
   * @param tensor_index Tensor index
   * @param handle Buffer handle to make active
   * @return kTfLiteOk on success
   */
  TfLiteStatus SetActiveBuffer(int tensor_index, TfLiteBufferHandle handle);

  /**
   * Get the currently active buffer handle for a tensor.
   * @param tensor_index Tensor index
   * @return Active handle, or kTfLiteNullBufferHandle if none set
   */
  TfLiteBufferHandle GetActiveBuffer(int tensor_index) const;

  /**
   * Get the fd for the currently active buffer for a tensor.
   * @param tensor_index Tensor index
   * @return fd on success, -1 if no active buffer
   */
  int GetActiveFd(int tensor_index) const;

  // === Queries ===

  /**
   * Copy scalar fields of an entry for a handle (excludes drm_attachment).
   * Returns true if found, false otherwise.
   * Thread-safe: copies scalar fields to avoid race conditions.
   */
  bool GetEntry(TfLiteBufferHandle handle, DmaBufEntry* out_entry) const;

  /**
   * Find handle by tensor index (returns first bound handle).
   * For buffer cycling, use GetActiveBuffer() instead.
   */
  TfLiteBufferHandle FindByTensorIndex(int tensor_index) const;

  /**
   * Get fd for a handle.
   */
  int GetFd(TfLiteBufferHandle handle) const;

  /**
   * Check if a tensor has any dmabuf bindings.
   */
  bool HasDmaBufBinding(int tensor_index) const;

 private:
  void SyncBuffer(int fd, bool start, bool for_write);
  int AllocateFromHeap(size_t size);
  std::unique_ptr<DrmAttachment> CreateDrmAttachment(int dmabuf_fd);

  int dma_heap_fd_;                                  // /dev/dma_heap fd
  std::map<TfLiteBufferHandle, DmaBufEntry> entries_;
  TfLiteBufferHandle next_handle_;
  mutable std::mutex mutex_;

  // Buffer cycling: maps tensor_index -> currently active handle
  std::map<int, TfLiteBufferHandle> active_buffers_;
};

}  // namespace delegate
}  // namespace vx

#endif  // TENSORFLOW_LITE_DELEGATES_VX_DELEGATE_DMABUF_MANAGER_H_
