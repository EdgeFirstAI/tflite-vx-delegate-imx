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

#include "hal_dmabuf.h"
#include "delegate_main.h"
#include "dmabuf_manager.h"
#include "vx_delegate_dmabuf.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "tim/vx/types.h"

namespace {

vx::delegate::DmaBufManager* GetDmaBufManager(TfLiteDelegate* delegate) {
  if (!delegate) return nullptr;
  auto* derived = reinterpret_cast<vx::delegate::DerivedDelegateData*>(delegate);
  if (!derived->enable_dmabuf || !derived->dmabuf_manager) {
    return nullptr;
  }
  return derived->dmabuf_manager.get();
}

TfLiteBufferHandle ResolveHandle(TfLiteDelegate* delegate, int tensor_index) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) return kTfLiteNullBufferHandle;

  TfLiteBufferHandle handle = mgr->GetActiveBuffer(tensor_index);
  if (handle == kTfLiteNullBufferHandle) {
    handle = mgr->FindByTensorIndex(tensor_index);
  }
  return handle;
}

// Maps a TIM-VX DataType to the corresponding hal_dtype enum value.
// Falls back to HAL_DTYPE_U8 for types that have no direct mapping (e.g.
// BOOL8, UNKNOWN) since those are not representable in hal_dtype.
hal_dtype TimVxDTypeToHalDType(tim::vx::DataType dt) {
  switch (dt) {
    case tim::vx::DataType::UINT8:   return HAL_DTYPE_U8;
    case tim::vx::DataType::INT8:    return HAL_DTYPE_I8;
    case tim::vx::DataType::UINT16:  return HAL_DTYPE_U16;
    case tim::vx::DataType::INT16:   return HAL_DTYPE_I16;
    case tim::vx::DataType::UINT32:  return HAL_DTYPE_U32;
    case tim::vx::DataType::INT32:   return HAL_DTYPE_I32;
    case tim::vx::DataType::INT64:   return HAL_DTYPE_I64;
    case tim::vx::DataType::FLOAT16: return HAL_DTYPE_F16;
    case tim::vx::DataType::FLOAT32: return HAL_DTYPE_F32;
    // BOOL8 and UNKNOWN have no hal_dtype equivalent; fall back to U8.
    default:                         return HAL_DTYPE_U8;
  }
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
hal_delegate_t hal_dmabuf_get_instance(void) {
  return static_cast<hal_delegate_t>(VxDelegateGetInstance());
}

__attribute__((visibility("default")))
int hal_dmabuf_is_supported(hal_delegate_t delegate) {
  auto* tfl_delegate = static_cast<TfLiteDelegate*>(delegate);
  return VxDelegateIsDmaBufSupported(tfl_delegate) ? 1 : 0;
}

__attribute__((visibility("default")))
int hal_dmabuf_get_tensor_info(hal_delegate_t delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info* info,
                               size_t info_size) {
  if (!delegate || !info || info_size < sizeof(hal_dmabuf_tensor_info)) {
    errno = EINVAL;
    return -1;
  }
  if (tensor_index < 0) {
    errno = EINVAL;
    return -1;
  }

  auto* tfl_delegate = static_cast<TfLiteDelegate*>(delegate);
  auto* mgr = GetDmaBufManager(tfl_delegate);
  if (!mgr) {
    errno = EIO;
    return -1;
  }

  TfLiteBufferHandle handle = ResolveHandle(tfl_delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) {
    errno = ERANGE;
    return -1;
  }

  vx::delegate::DmaBufEntry entry;
  if (!mgr->GetEntry(handle, &entry)) {
    errno = EIO;
    return -1;
  }

  std::memset(info, 0, info_size);
  info->fd = entry.fd;
  info->offset = 0;
  info->size = entry.size;

  // Populate shape, ndim, and dtype from the TIM-VX tensor when available.
  // The tensor is created lazily (on first graph compilation), so it may be
  // null if get_tensor_info is called before the first Invoke().
  if (entry.tim_tensor) {
    const tim::vx::ShapeType& shape = entry.tim_tensor->GetShape();
    size_t ndim = std::min(shape.size(), static_cast<size_t>(HAL_DMABUF_MAX_NDIM));
    info->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
      info->shape[i] = static_cast<size_t>(shape[i]);
    }
    info->dtype = TimVxDTypeToHalDType(entry.tim_tensor->GetDataType());
  }
  // If tim_tensor is null the fields remain zero-initialised (ndim=0,
  // dtype=HAL_DTYPE_U8) from the memset above.

  return 0;
}

__attribute__((visibility("default")))
int hal_dmabuf_sync_for_device(hal_delegate_t delegate, int tensor_index) {
  if (!delegate) {
    errno = EINVAL;
    return -1;
  }
  if (tensor_index < 0) {
    errno = EINVAL;
    return -1;
  }

  auto* tfl_delegate = static_cast<TfLiteDelegate*>(delegate);
  if (!GetDmaBufManager(tfl_delegate)) {
    errno = EIO;
    return -1;
  }

  TfLiteBufferHandle handle = ResolveHandle(tfl_delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) {
    errno = ERANGE;
    return -1;
  }

  if (VxDelegateSyncForDevice(tfl_delegate, handle) != kTfLiteOk) {
    errno = EIO;
    return -1;
  }
  return 0;
}

__attribute__((visibility("default")))
int hal_dmabuf_sync_for_cpu(hal_delegate_t delegate, int tensor_index) {
  if (!delegate) {
    errno = EINVAL;
    return -1;
  }
  if (tensor_index < 0) {
    errno = EINVAL;
    return -1;
  }

  auto* tfl_delegate = static_cast<TfLiteDelegate*>(delegate);
  if (!GetDmaBufManager(tfl_delegate)) {
    errno = EIO;
    return -1;
  }

  TfLiteBufferHandle handle = ResolveHandle(tfl_delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) {
    errno = ERANGE;
    return -1;
  }

  if (VxDelegateSyncForCpu(tfl_delegate, handle) != kTfLiteOk) {
    errno = EIO;
    return -1;
  }
  return 0;
}

__attribute__((visibility("default")))
int hal_camera_adaptor_is_supported(hal_delegate_t delegate,
                                    const char* format) {
  (void)delegate;
  if (!format) return 0;
  return VxCameraAdaptorIsSupported(format) ? 1 : 0;
}

__attribute__((visibility("default")))
int hal_camera_adaptor_get_format_info(hal_delegate_t delegate,
                                       const char* format,
                                       hal_camera_adaptor_format_info* info,
                                       size_t info_size) {
  (void)delegate;
  if (!format || !info || info_size < sizeof(hal_camera_adaptor_format_info)) {
    errno = EINVAL;
    return -1;
  }

  if (!VxCameraAdaptorIsSupported(format)) {
    errno = ENOTSUP;
    return -1;
  }

  std::memset(info, 0, info_size);
  info->input_channels = VxCameraAdaptorGetInputChannels(format);
  info->output_channels = VxCameraAdaptorGetOutputChannels(format);

  const char* fourcc = VxCameraAdaptorGetFourCC(format);
  if (fourcc) {
    std::strncpy(info->fourcc, fourcc, sizeof(info->fourcc) - 1);
    info->fourcc[sizeof(info->fourcc) - 1] = '\0';
  }

  return 0;
}

}  // extern "C"
