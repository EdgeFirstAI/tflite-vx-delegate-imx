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
#include <cstring>

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

}  // namespace

extern "C" {

bool hal_dmabuf_is_supported(TfLiteDelegate* delegate) {
  return VxDelegateIsDmaBufSupported(delegate);
}

TfLiteDelegate* hal_dmabuf_get_instance(void) {
  return VxDelegateGetInstance();
}

int hal_dmabuf_get_tensor_info(TfLiteDelegate* delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info* info,
                               size_t info_size) {
  if (!info || info_size == 0) return -1;

  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) return -1;

  TfLiteBufferHandle handle = ResolveHandle(delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) return -1;

  vx::delegate::DmaBufEntry entry;
  if (!mgr->GetEntry(handle, &entry)) return -1;

  hal_dmabuf_tensor_info full_info = {};
  full_info.fd = entry.fd;
  full_info.offset = 0;
  full_info.size = entry.size;

  std::memcpy(info, &full_info, std::min(info_size, sizeof(full_info)));
  return 0;
}

int hal_dmabuf_sync_for_device(TfLiteDelegate* delegate, int tensor_index) {
  if (!GetDmaBufManager(delegate)) return -1;

  TfLiteBufferHandle handle = ResolveHandle(delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) return -1;

  return VxDelegateSyncForDevice(delegate, handle) == kTfLiteOk ? 0 : -1;
}

int hal_dmabuf_sync_for_cpu(TfLiteDelegate* delegate, int tensor_index) {
  if (!GetDmaBufManager(delegate)) return -1;

  TfLiteBufferHandle handle = ResolveHandle(delegate, tensor_index);
  if (handle == kTfLiteNullBufferHandle) return -1;

  return VxDelegateSyncForCpu(delegate, handle) == kTfLiteOk ? 0 : -1;
}

}  // extern "C"
