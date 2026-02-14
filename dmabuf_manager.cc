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

#include "dmabuf_manager.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <cerrno>
#include <cstring>
#include <cstdint>

#include "tensorflow/lite/minimal_logging.h"

namespace vx {
namespace delegate {

// =============================================================================
// DRM ioctl definitions for PRIME import (avoids libdrm dependency)
// =============================================================================
//
// The CMA heap's begin_cpu_access iterates over buffer->attachments to perform
// cache maintenance via dma_sync_sgtable_for_cpu(). Without any active
// attachments, DMA_BUF_IOCTL_SYNC is a no-op.
//
// By importing the DMA-buf fd through the DRM/GPU driver
// (DRM_IOCTL_PRIME_FD_TO_HANDLE), the driver creates a persistent
// dma_buf_attach(). This makes DMA_BUF_IOCTL_SYNC effective.

namespace {

struct DrmPrimeHandle {
  uint32_t handle;
  uint32_t flags;
  int32_t fd;
};

struct DrmGemClose {
  uint32_t handle;
  uint32_t pad;
};

// DRM_IOCTL_PRIME_FD_TO_HANDLE = _IOWR('d', 0x2e, struct drm_prime_handle)
constexpr unsigned long kDrmIoctlPrimeFdToHandle =
    (3UL << 30) | (sizeof(DrmPrimeHandle) << 16) | ('d' << 8) | 0x2e;

// DRM_IOCTL_GEM_CLOSE = _IOW('d', 0x09, struct drm_gem_close)
constexpr unsigned long kDrmIoctlGemClose =
    (1UL << 30) | (sizeof(DrmGemClose) << 16) | ('d' << 8) | 0x09;

const char* kDrmRenderNode = "/dev/dri/renderD128";

// Default paths for dma_heap devices
const char* kDmaHeapPaths[] = {
    "/dev/dma_heap/linux,cma",    // Preferred: CMA heap for NPU
    "/dev/dma_heap/system",        // Fallback: system heap
    nullptr
};

}  // namespace

// =============================================================================
// DrmAttachment implementation
// =============================================================================

DrmAttachment::DrmAttachment(int drm_fd, uint32_t gem_handle)
    : drm_fd_(drm_fd), gem_handle_(gem_handle) {}

DrmAttachment::~DrmAttachment() {
  DrmGemClose close_arg = {};
  close_arg.handle = gem_handle_;
  ioctl(drm_fd_, kDrmIoctlGemClose, &close_arg);
  close(drm_fd_);
}

std::unique_ptr<DrmAttachment> DrmAttachment::Create(int dmabuf_fd) {
  int drm_fd = open(kDrmRenderNode, O_RDWR | O_CLOEXEC);
  if (drm_fd < 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "DrmAttachment: %s not available: %s",
                    kDrmRenderNode, strerror(errno));
    return nullptr;
  }

  DrmPrimeHandle prime = {};
  prime.fd = dmabuf_fd;

  if (ioctl(drm_fd, kDrmIoctlPrimeFdToHandle, &prime) < 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "DrmAttachment: PRIME_FD_TO_HANDLE failed for fd=%d: %s",
                    dmabuf_fd, strerror(errno));
    close(drm_fd);
    return nullptr;
  }

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DrmAttachment: imported fd=%d as GEM handle %u",
                  dmabuf_fd, prime.handle);

  return std::unique_ptr<DrmAttachment>(
      new DrmAttachment(drm_fd, prime.handle));
}

DmaBufManager::DmaBufManager()
    : dma_heap_fd_(-1), next_handle_(1) {}

DmaBufManager::~DmaBufManager() {
  // Close delegate-owned buffers
  for (auto& [handle, entry] : entries_) {
    if (entry.ownership == kVxDmaBufOwnerDelegate && entry.fd >= 0) {
      close(entry.fd);
    }
  }
  entries_.clear();

  if (dma_heap_fd_ >= 0) {
    close(dma_heap_fd_);
    dma_heap_fd_ = -1;
  }
}

bool DmaBufManager::Initialize(const char* heap_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (dma_heap_fd_ >= 0) {
    return true;  // Already initialized
  }

  // Try specified path first, then defaults
  if (heap_path && heap_path[0] != '\0') {
    dma_heap_fd_ = open(heap_path, O_RDWR);
    if (dma_heap_fd_ >= 0) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "DmaBufManager: opened %s", heap_path);
      return true;
    }
  }

  // Try default paths
  for (const char** path = kDmaHeapPaths; *path != nullptr; ++path) {
    dma_heap_fd_ = open(*path, O_RDWR);
    if (dma_heap_fd_ >= 0) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "DmaBufManager: opened %s", *path);
      return true;
    }
  }

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                  "DmaBufManager: no dma_heap device available");
  return false;
}

TfLiteBufferHandle DmaBufManager::RegisterBuffer(int fd, size_t size,
                                                  VxDmaBufSyncMode sync_mode) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (fd < 0 || size == 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "DmaBufManager::RegisterBuffer: invalid fd=%d size=%zu",
                    fd, size);
    return kTfLiteNullBufferHandle;
  }

  TfLiteBufferHandle handle = next_handle_++;

  DmaBufEntry entry;
  entry.fd = fd;
  entry.size = size;
  entry.ownership = kVxDmaBufOwnerClient;
  entry.default_sync = sync_mode;
  entry.tensor_index = -1;
  entry.tim_tensor = nullptr;
  entry.is_input = false;
  entry.is_output = false;
  entry.drm_attachment = CreateDrmAttachment(fd);

  entries_[handle] = std::move(entry);

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DmaBufManager: registered fd=%d size=%zu as handle=%d",
                  fd, size, handle);

  return handle;
}

TfLiteStatus DmaBufManager::UnregisterBuffer(TfLiteBufferHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return kTfLiteError;
  }

  // Don't close fd for client-owned buffers
  if (it->second.ownership == kVxDmaBufOwnerDelegate && it->second.fd >= 0) {
    close(it->second.fd);
  }

  entries_.erase(it);
  return kTfLiteOk;
}

int DmaBufManager::AllocateFromHeap(size_t size) {
  if (dma_heap_fd_ < 0) {
    return -1;
  }

  // Round up to 4KB page boundary for NPU alignment requirements
  const size_t page_size = 4096;
  size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

  struct dma_heap_allocation_data alloc_data = {};
  alloc_data.len = aligned_size;
  alloc_data.fd_flags = O_RDWR | O_CLOEXEC;

  if (ioctl(dma_heap_fd_, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "DmaBufManager: DMA_HEAP_IOCTL_ALLOC failed: %s",
                    strerror(errno));
    return -1;
  }

  return alloc_data.fd;
}

TfLiteBufferHandle DmaBufManager::AllocateBuffer(size_t size, size_t alignment,
                                                  VxDmaBufOwnership ownership,
                                                  VxDmaBufDesc* desc) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (size == 0) {
    return kTfLiteNullBufferHandle;
  }

  // Round up to alignment if specified
  if (alignment > 0) {
    size = (size + alignment - 1) & ~(alignment - 1);
  }

  int fd = AllocateFromHeap(size);
  if (fd < 0) {
    return kTfLiteNullBufferHandle;
  }

  TfLiteBufferHandle handle = next_handle_++;

  DmaBufEntry entry;
  entry.fd = fd;
  entry.size = size;
  entry.ownership = ownership;
  entry.default_sync = kVxDmaBufSyncReadWrite;
  entry.tensor_index = -1;
  entry.tim_tensor = nullptr;
  entry.is_input = false;
  entry.is_output = false;
  entry.drm_attachment = CreateDrmAttachment(fd);

  entries_[handle] = std::move(entry);

  if (desc) {
    desc->fd = fd;
    desc->size = size;
    desc->map_ptr = nullptr;
  }

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DmaBufManager: allocated fd=%d size=%zu as handle=%d",
                  fd, size, handle);

  return handle;
}

TfLiteStatus DmaBufManager::ReleaseBuffer(TfLiteBufferHandle handle) {
  return UnregisterBuffer(handle);
}

TfLiteStatus DmaBufManager::BindToTensor(TfLiteBufferHandle handle,
                                          int tensor_index,
                                          bool is_input, bool is_output) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return kTfLiteError;
  }

  it->second.tensor_index = tensor_index;
  it->second.is_input = is_input;
  it->second.is_output = is_output;

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DmaBufManager: bound handle=%d to tensor=%d (input=%d, output=%d)",
                  handle, tensor_index, is_input, is_output);

  return kTfLiteOk;
}

std::shared_ptr<tim::vx::Tensor> DmaBufManager::GetOrCreateTensor(
    TfLiteBufferHandle handle,
    std::shared_ptr<tim::vx::Graph>& graph,
    const tim::vx::TensorSpec& spec) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return nullptr;
  }

  // Return existing tensor if already created
  if (it->second.tim_tensor) {
    return it->second.tim_tensor;
  }

  // Create TIM-VX tensor with DMA-BUF descriptor
  tim::vx::DmaBufferDesc dma_desc;
  dma_desc.fd = it->second.fd;

  auto tensor = graph->CreateTensor(spec, dma_desc);
  if (!tensor) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "DmaBufManager: failed to create TIM-VX tensor with dmabuf fd=%d",
                    it->second.fd);
    return nullptr;
  }

  it->second.tim_tensor = tensor;

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DmaBufManager: created TIM-VX tensor for handle=%d fd=%d",
                  handle, it->second.fd);

  return tensor;
}

void DmaBufManager::UpdateTensorAfterLayoutInference(
    TfLiteBufferHandle handle,
    std::shared_ptr<tim::vx::Tensor> inferred_tensor) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it != entries_.end() && inferred_tensor) {
    it->second.tim_tensor = inferred_tensor;
  }
}

void DmaBufManager::SyncBuffer(int fd, bool start, bool for_write) {
  struct dma_buf_sync sync = {};

  sync.flags = start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END;
  sync.flags |= for_write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;

  if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "DmaBufManager: DMA_BUF_IOCTL_SYNC failed: %s",
                    strerror(errno));
  }
}

TfLiteStatus DmaBufManager::BeginCpuAccess(TfLiteBufferHandle handle,
                                            VxDmaBufSyncMode mode) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return kTfLiteError;
  }

  bool for_write = (mode == kVxDmaBufSyncWrite ||
                    mode == kVxDmaBufSyncReadWrite);
  SyncBuffer(it->second.fd, /*start=*/true, for_write);

  return kTfLiteOk;
}

TfLiteStatus DmaBufManager::EndCpuAccess(TfLiteBufferHandle handle,
                                          VxDmaBufSyncMode mode) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return kTfLiteError;
  }

  bool for_write = (mode == kVxDmaBufSyncWrite ||
                    mode == kVxDmaBufSyncReadWrite);
  SyncBuffer(it->second.fd, /*start=*/false, for_write);

  return kTfLiteOk;
}

bool DmaBufManager::GetEntry(TfLiteBufferHandle handle, DmaBufEntry* out_entry) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return false;
  }
  if (out_entry) {
    out_entry->fd = it->second.fd;
    out_entry->size = it->second.size;
    out_entry->ownership = it->second.ownership;
    out_entry->default_sync = it->second.default_sync;
    out_entry->tensor_index = it->second.tensor_index;
    out_entry->tim_tensor = it->second.tim_tensor;
    out_entry->is_input = it->second.is_input;
    out_entry->is_output = it->second.is_output;
    // drm_attachment is not copied — it's internal to the manager
  }
  return true;
}

TfLiteBufferHandle DmaBufManager::FindByTensorIndex(int tensor_index) const {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& [handle, entry] : entries_) {
    if (entry.tensor_index == tensor_index) {
      return handle;
    }
  }
  return kTfLiteNullBufferHandle;
}

int DmaBufManager::GetFd(TfLiteBufferHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return -1;
  }
  return it->second.fd;
}

TfLiteStatus DmaBufManager::SetActiveBuffer(int tensor_index,
                                             TfLiteBufferHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Verify handle exists and is bound to this tensor
  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "DmaBufManager::SetActiveBuffer: handle %d not found",
                    handle);
    return kTfLiteError;
  }

  if (it->second.tensor_index != tensor_index) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "DmaBufManager::SetActiveBuffer: handle %d bound to tensor %d, not %d",
                    handle, it->second.tensor_index, tensor_index);
    return kTfLiteError;
  }

  active_buffers_[tensor_index] = handle;

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "DmaBufManager: set active buffer for tensor %d to handle %d (fd=%d)",
                  tensor_index, handle, it->second.fd);

  return kTfLiteOk;
}

TfLiteBufferHandle DmaBufManager::GetActiveBuffer(int tensor_index) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = active_buffers_.find(tensor_index);
  if (it != active_buffers_.end()) {
    return it->second;
  }

  // Fallback: if no active buffer set, return first bound handle
  for (const auto& [handle, entry] : entries_) {
    if (entry.tensor_index == tensor_index) {
      return handle;
    }
  }
  return kTfLiteNullBufferHandle;
}

int DmaBufManager::GetActiveFd(int tensor_index) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto active_it = active_buffers_.find(tensor_index);
  if (active_it != active_buffers_.end()) {
    auto entry_it = entries_.find(active_it->second);
    if (entry_it != entries_.end()) {
      return entry_it->second.fd;
    }
  }

  // Fallback: if no active buffer set, return fd of first bound handle
  for (const auto& [handle, entry] : entries_) {
    if (entry.tensor_index == tensor_index) {
      return entry.fd;
    }
  }
  return -1;
}

bool DmaBufManager::HasDmaBufBinding(int tensor_index) const {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& [handle, entry] : entries_) {
    if (entry.tensor_index == tensor_index) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<DrmAttachment> DmaBufManager::CreateDrmAttachment(int dmabuf_fd) {
  auto attachment = DrmAttachment::Create(dmabuf_fd);
  if (!attachment) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "DmaBufManager: DRM attachment failed for fd=%d; "
                    "DMA_BUF_IOCTL_SYNC may be a no-op on cached CMA heaps",
                    dmabuf_fd);
  }
  return attachment;
}

}  // namespace delegate
}  // namespace vx
