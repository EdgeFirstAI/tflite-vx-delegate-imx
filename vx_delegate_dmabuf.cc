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

#include "vx_delegate_dmabuf.h"
#include "delegate_main.h"

namespace {

// Get the DmaBufManager from the TfLiteDelegate
vx::delegate::DmaBufManager* GetDmaBufManager(TfLiteDelegate* delegate) {
  if (!delegate) return nullptr;
  
  auto* derived = reinterpret_cast<vx::delegate::DerivedDelegateData*>(delegate);
  if (!derived->enable_dmabuf || !derived->dmabuf_manager) {
    return nullptr;
  }
  return derived->dmabuf_manager.get();
}

bool IsDmaBufEnabled(TfLiteDelegate* delegate) {
  if (!delegate) return false;
  auto* derived = reinterpret_cast<vx::delegate::DerivedDelegateData*>(delegate);
  return derived->enable_dmabuf && derived->dmabuf_manager && 
         derived->dmabuf_manager->IsSupported();
}

}  // namespace

extern "C" {

TfLiteBufferHandle VxDelegateRegisterDmaBuf(TfLiteDelegate* delegate,
                                             int fd,
                                             size_t size,
                                             VxDmaBufSyncMode sync_mode) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteNullBufferHandle;
  }
  return mgr->RegisterBuffer(fd, size, sync_mode);
}

TfLiteStatus VxDelegateUnregisterDmaBuf(TfLiteDelegate* delegate,
                                         TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  return mgr->UnregisterBuffer(handle);
}

TfLiteBufferHandle VxDelegateRequestDmaBuf(TfLiteDelegate* delegate,
                                            int tensor_index,
                                            VxDmaBufOwnership ownership,
                                            VxDmaBufDesc* desc) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr || !desc) {
    return kTfLiteNullBufferHandle;
  }

  // Note: This function expects the caller to have already determined the
  // required buffer size (e.g., from interpreter->tensor(idx)->bytes after
  // AllocateTensors() has been called).
  //
  // If size is not known, the caller should:
  // 1. Call interpreter->AllocateTensors() first
  // 2. Query tensor size with interpreter->tensor(tensor_index)->bytes
  // 3. Call this function with the correct size in desc->size
  //
  // For convenience, if desc->size is 0, this function will fail.
  if (desc->size == 0) {
    return kTfLiteNullBufferHandle;
  }

  // Allocate buffer from DMA heap
  TfLiteBufferHandle handle = mgr->AllocateBuffer(
      desc->size,
      64,  // 64-byte alignment for NPU
      ownership,
      desc);

  if (handle == kTfLiteNullBufferHandle) {
    return kTfLiteNullBufferHandle;
  }

  // Bind to the specified tensor
  if (tensor_index >= 0) {
    // Determine if this is input or output based on tensor_index
    // For now, assume both (the actual binding happens during graph modification)
    if (mgr->BindToTensor(handle, tensor_index, true, true) != kTfLiteOk) {
      // Cleanup on bind failure
      mgr->ReleaseBuffer(handle);
      return kTfLiteNullBufferHandle;
    }
  }

  return handle;
}

TfLiteStatus VxDelegateReleaseDmaBuf(TfLiteDelegate* delegate,
                                      TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  return mgr->ReleaseBuffer(handle);
}

TfLiteStatus VxDelegateBeginCpuAccess(TfLiteDelegate* delegate,
                                       TfLiteBufferHandle handle,
                                       VxDmaBufSyncMode mode) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  return mgr->BeginCpuAccess(handle, mode);
}

TfLiteStatus VxDelegateEndCpuAccess(TfLiteDelegate* delegate,
                                     TfLiteBufferHandle handle,
                                     VxDmaBufSyncMode mode) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  return mgr->EndCpuAccess(handle, mode);
}

TfLiteStatus VxDelegateBindDmaBufToTensor(TfLiteDelegate* delegate,
                                           TfLiteBufferHandle handle,
                                           int tensor_index) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  // For now, assume input. The direction is determined by whether the tensor
  // is in subgraph_inputs or subgraph_outputs during Invoke.
  return mgr->BindToTensor(handle, tensor_index, true, true);
}

bool VxDelegateIsDmaBufSupported(TfLiteDelegate* delegate) {
  return IsDmaBufEnabled(delegate);
}

int VxDelegateGetDmaBufFd(TfLiteDelegate* delegate, TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return -1;
  }
  return mgr->GetFd(handle);
}

TfLiteStatus VxDelegateSyncForDevice(TfLiteDelegate* delegate,
                                      TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  // End CPU write access = flush caches for device
  return mgr->EndCpuAccess(handle, kVxDmaBufSyncWrite);
}

TfLiteStatus VxDelegateSyncForCpu(TfLiteDelegate* delegate,
                                   TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  // Begin CPU read access = invalidate caches
  return mgr->BeginCpuAccess(handle, kVxDmaBufSyncRead);
}

TfLiteStatus VxDelegateSetActiveDmaBuf(TfLiteDelegate* delegate,
                                        int tensor_index,
                                        TfLiteBufferHandle handle) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteError;
  }
  return mgr->SetActiveBuffer(tensor_index, handle);
}

TfLiteStatus VxDelegateInvalidateGraph(TfLiteDelegate* delegate) {
  if (!delegate) {
    return kTfLiteError;
  }

  auto* derived = reinterpret_cast<vx::delegate::DerivedDelegateData*>(delegate);
  derived->needs_invalidation = true;

  return kTfLiteOk;
}

bool VxDelegateIsGraphCompiled(TfLiteDelegate* delegate) {
  if (!delegate) {
    return false;
  }

  // Note: This checks the invalidation flag, not the actual compiled state
  // of individual Delegate instances (which are per-node).
  // After invalidation, the next Invoke() will recompile.
  auto* derived = reinterpret_cast<vx::delegate::DerivedDelegateData*>(delegate);
  return !derived->needs_invalidation;
}

TfLiteBufferHandle VxDelegateGetActiveBuffer(TfLiteDelegate* delegate,
                                              int tensor_index) {
  auto* mgr = GetDmaBufManager(delegate);
  if (!mgr) {
    return kTfLiteNullBufferHandle;
  }
  return mgr->GetActiveBuffer(tensor_index);
}

}  // extern "C"
