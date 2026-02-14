/* Copyright 2025 Au-Zone Technologies
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 *
 * DMA-BUF Zero-Copy Test and Benchmark for TFLite VX Delegate
 *
 * This example demonstrates and validates the zero-copy inference path using
 * Linux DMA-BUF. It runs the same model through both copy-based and zero-copy
 * paths, verifies the outputs match, and reports timing statistics.
 *
 * Zero-copy eliminates memcpy operations between CPU and NPU by sharing
 * DMA-BUF file descriptors. This is essential for efficient camera/video
 * pipelines where V4L2 can write directly to buffers the NPU reads from.
 *
 * Usage:
 *   dmabuf_benchmark <model.tflite> [--iterations N] [--warmup N] [--test]
 *                    [--dma-heap <device>]
 *
 * Modes:
 *   Default:  Benchmark mode - 100 iterations with timing statistics
 *   --test:   Test mode - 1 iteration with detailed output comparison
 *
 * Exit codes:
 *   0: Success (outputs match between copy and zero-copy paths)
 *   1: Failure (outputs differ or error occurred)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// TensorFlow Lite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// VX Delegate headers
#include "delegate_main.h"
#include "vx_delegate_dmabuf.h"
#include "vsi_npu_custom_op.h"

// Linux DMA-BUF headers for buffer allocation and cache synchronization
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

namespace {

// ----------------------------------------------------------------------------
// Timing Statistics
// ----------------------------------------------------------------------------

struct Stats {
  std::vector<double> samples;

  void add(double v) { samples.push_back(v); }
  void clear() { samples.clear(); }

  double mean() const {
    if (samples.empty()) return 0;
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  }

  double p50() const { return percentile(50); }
  double p99() const { return percentile(99); }

  double percentile(double p) const {
    if (samples.empty()) return 0;
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = std::min(size_t(p / 100.0 * sorted.size()), sorted.size() - 1);
    return sorted[idx];
  }
};

// ----------------------------------------------------------------------------
// DMA-BUF Allocation
// ----------------------------------------------------------------------------

// Default dma_heap device - CMA provides cached contiguous memory that the NPU
// can access directly. This is the preferred heap for zero-copy inference.
constexpr const char* kDefaultDmaHeapDevice = "/dev/dma_heap/linux,cma";

// DmaHeap wraps the Linux dma_heap interface for allocating DMA-capable buffers.
// The device path is configurable to support different dma_heap providers.
class DmaHeap {
 public:
  explicit DmaHeap(const char* device = kDefaultDmaHeapDevice) : device_(device) {
    fd_ = open(device, O_RDWR);
    if (fd_ < 0) {
      std::cerr << "Failed to open dma_heap device: " << device << "\n";
    }
  }
  ~DmaHeap() { if (fd_ >= 0) close(fd_); }

  const char* device() const { return device_; }

  bool ok() const { return fd_ >= 0; }

  // Allocate a DMA buffer of the given size, returns the dmabuf file descriptor
  int alloc(size_t size) {
    if (fd_ < 0) return -1;
    struct dma_heap_allocation_data data = {};
    data.len = (size + 4095) & ~4095;  // Page-align for DMA requirements
    data.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(fd_, DMA_HEAP_IOCTL_ALLOC, &data) < 0) return -1;
    return data.fd;
  }

 private:
  const char* device_;
  int fd_ = -1;
};

// ----------------------------------------------------------------------------
// DMA-BUF Wrapper
// ----------------------------------------------------------------------------

// DmaBuf wraps a dmabuf file descriptor with RAII semantics and provides
// convenient methods for memory mapping and cache synchronization.
//
// Cache synchronization is required because CMA memory is cached:
// - sync_start_write/sync_end_write: Bracket CPU writes to the buffer
// - sync_start_read/sync_end_read: Bracket CPU reads from the buffer
//
// For H2H (hardware-to-hardware) pipelines where CPU never touches the data,
// no synchronization is needed.
class DmaBuf {
 public:
  DmaBuf() = default;
  DmaBuf(int fd, size_t size) : fd_(fd), size_(size) {}
  ~DmaBuf() { reset(); }

  // Move semantics - dmabuf ownership transfers
  DmaBuf(DmaBuf&& o) noexcept : fd_(o.fd_), size_(o.size_), ptr_(o.ptr_) {
    o.fd_ = -1; o.ptr_ = nullptr;
  }
  DmaBuf& operator=(DmaBuf&& o) noexcept {
    if (this != &o) {
      reset();
      fd_ = o.fd_; size_ = o.size_; ptr_ = o.ptr_;
      o.fd_ = -1; o.ptr_ = nullptr;
    }
    return *this;
  }

  // No copying - each dmabuf fd should have single owner
  DmaBuf(const DmaBuf&) = delete;
  DmaBuf& operator=(const DmaBuf&) = delete;

  void reset() {
    if (ptr_) { munmap(ptr_, size_); ptr_ = nullptr; }
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
  }

  bool ok() const { return fd_ >= 0; }
  int fd() const { return fd_; }
  size_t size() const { return size_; }

  // Memory-map the buffer for CPU access. The mapping persists until reset().
  void* map() {
    if (!ptr_ && fd_ >= 0) {
      ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      if (ptr_ == MAP_FAILED) ptr_ = nullptr;
    }
    return ptr_;
  }

  // Cache sync for CPU write access (invalidate before, flush after)
  void sync_start_write() {
    struct dma_buf_sync s = {DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE};
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }
  void sync_end_write() {
    struct dma_buf_sync s = {DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }

  // Cache sync for CPU read access (invalidate before, end after)
  void sync_start_read() {
    struct dma_buf_sync s = {DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }
  void sync_end_read() {
    struct dma_buf_sync s = {DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ};
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  void* ptr_ = nullptr;
};

// ----------------------------------------------------------------------------
// Timer
// ----------------------------------------------------------------------------

class Timer {
 public:
  void start() { t_ = std::chrono::high_resolution_clock::now(); }
  double us() const {
    return std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t_).count();
  }
 private:
  std::chrono::high_resolution_clock::time_point t_;
};

// ----------------------------------------------------------------------------
// Interpreter Creation
// ----------------------------------------------------------------------------

// Create a TFLite interpreter with the VX Delegate.
// The dmabuf parameter enables the zero-copy API on the delegate.
std::unique_ptr<tflite::Interpreter> make_interpreter(
    tflite::FlatBufferModel* model, TfLiteDelegate*& delegate, bool dmabuf) {
  vx::delegate::VxDelegateOptions opts = vx::delegate::VxDelegateOptionsDefault();
  opts.enable_dmabuf = dmabuf;
  delegate = vx::delegate::VxDelegateCreate(&opts);
  if (!delegate) return nullptr;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  // Register the custom op for pre-compiled NPU graphs
  resolver.AddCustom(kNbgCustomOp, tflite::ops::custom::Register_VSI_NPU_PRECOMPILED());

  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder(*model, resolver)(&interp);
  if (!interp) return nullptr;

  if (interp->ModifyGraphWithDelegate(delegate) != kTfLiteOk ||
      interp->AllocateTensors() != kTfLiteOk) {
    vx::delegate::VxDelegateDelete(delegate);
    delegate = nullptr;
    return nullptr;
  }
  return interp;
}

// ----------------------------------------------------------------------------
// Output Comparison
// ----------------------------------------------------------------------------

// Compare two buffers and return the maximum absolute difference.
// Used to verify that zero-copy produces the same results as copy-based.
template<typename T>
double max_diff(const T* a, const T* b, size_t n) {
  double maxd = 0;
  for (size_t i = 0; i < n; i++) {
    double d = std::abs(double(a[i]) - double(b[i]));
    if (d > maxd) maxd = d;
  }
  return maxd;
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.tflite> [options]\n"
              << "Options:\n"
              << "  --iterations N   Number of benchmark iterations (default: 100)\n"
              << "  --warmup N       Warmup iterations (default: 10)\n"
              << "  --test           Test mode: 1 iteration, detailed comparison\n"
              << "  --dma-heap PATH  DMA heap device (default: " << kDefaultDmaHeapDevice << ")\n";
    return 1;
  }

  const char* model_path = argv[1];
  const char* dma_heap_device = kDefaultDmaHeapDevice;
  int iterations = 100;
  int warmup = 10;
  bool test_mode = false;

  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--test")) { test_mode = true; iterations = 1; warmup = 1; }
    else if (!strcmp(argv[i], "--dma-heap") && i + 1 < argc) dma_heap_device = argv[++i];
  }

  // -------------------------------------------------------------------------
  // Load Model
  // -------------------------------------------------------------------------

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model) {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  std::cout << "=== DMA-BUF Zero-Copy " << (test_mode ? "Test" : "Benchmark") << " ===\n";
  std::cout << "Model: " << model_path << "\n";
  if (!test_mode) std::cout << "Iterations: " << iterations << ", Warmup: " << warmup << "\n";

  // -------------------------------------------------------------------------
  // Create Interpreters
  // -------------------------------------------------------------------------

  // We create two interpreters from the same model:
  // 1. copy_interp: Uses the standard copy-based path (reference)
  // 2. zc_interp: Uses the zero-copy path with dmabuf

  TfLiteDelegate* copy_del = nullptr;
  auto copy_interp = make_interpreter(model.get(), copy_del, false);
  if (!copy_interp) {
    std::cerr << "Failed to create copy-based interpreter\n";
    return 1;
  }

  TfLiteDelegate* zc_del = nullptr;
  auto zc_interp = make_interpreter(model.get(), zc_del, true);
  if (!zc_interp) {
    std::cerr << "Failed to create zero-copy interpreter\n";
    vx::delegate::VxDelegateDelete(copy_del);
    return 1;
  }

  // -------------------------------------------------------------------------
  // Get Tensor Information
  // -------------------------------------------------------------------------

  auto inputs = copy_interp->inputs();
  auto outputs = copy_interp->outputs();
  size_t in_bytes = 0, out_bytes = 0;
  for (int i : inputs) in_bytes += copy_interp->tensor(i)->bytes;
  for (int i : outputs) out_bytes += copy_interp->tensor(i)->bytes;

  std::cout << "Input: " << in_bytes << " bytes (" << inputs.size() << " tensor"
            << (inputs.size() > 1 ? "s" : "") << ")\n";
  std::cout << "Output: " << out_bytes << " bytes (" << outputs.size() << " tensor"
            << (outputs.size() > 1 ? "s" : "") << ")\n";

  // -------------------------------------------------------------------------
  // Allocate and Register DMA Buffers
  // -------------------------------------------------------------------------

  // For zero-copy, we allocate dmabufs and register them with the delegate.
  // In a real application, these would come from V4L2 (camera) or DRM (display).

  DmaHeap heap(dma_heap_device);
  std::vector<DmaBuf> in_bufs, out_bufs;
  bool dmabuf_ok = heap.ok() && VxDelegateIsDmaBufSupported(zc_del);

  if (dmabuf_ok) {
    // Allocate and register input dmabufs
    for (int idx : inputs) {
      size_t sz = zc_interp->tensor(idx)->bytes;
      int fd = heap.alloc(sz);
      if (fd < 0) { dmabuf_ok = false; break; }
      in_bufs.emplace_back(fd, sz);

      // Register the dmabuf with the delegate and bind to tensor index
      auto h = VxDelegateRegisterDmaBuf(zc_del, fd, sz, kVxDmaBufSyncNone);
      VxDelegateBindDmaBufToTensor(zc_del, h, idx);
    }

    // Allocate and register output dmabufs
    for (int idx : outputs) {
      size_t sz = zc_interp->tensor(idx)->bytes;
      int fd = heap.alloc(sz);
      if (fd < 0) { dmabuf_ok = false; break; }
      out_bufs.emplace_back(fd, sz);

      auto h = VxDelegateRegisterDmaBuf(zc_del, fd, sz, kVxDmaBufSyncNone);
      VxDelegateBindDmaBufToTensor(zc_del, h, idx);
    }
  }

  if (!dmabuf_ok) {
    std::cout << "\nDMA-BUF not available - cannot test zero-copy path\n";
    vx::delegate::VxDelegateDelete(copy_del);
    vx::delegate::VxDelegateDelete(zc_del);
    return 1;
  }
  std::cout << "DMA heap: " << heap.device() << "\n\n";

  // -------------------------------------------------------------------------
  // Prepare Input Data
  // -------------------------------------------------------------------------

  // Generate deterministic random input for reproducibility
  std::vector<uint8_t> input_data(in_bytes);
  srand(42);
  for (auto& b : input_data) b = rand() % 256;

  // Copy input to copy-based interpreter's tensor
  {
    size_t off = 0;
    for (int idx : inputs) {
      size_t sz = copy_interp->tensor(idx)->bytes;
      memcpy(copy_interp->tensor(idx)->data.raw, input_data.data() + off, sz);
      off += sz;
    }
  }

  // Copy input to zero-copy interpreter's dmabuf (with proper cache sync)
  {
    size_t off = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
      size_t sz = in_bufs[i].size();
      void* ptr = in_bufs[i].map();
      in_bufs[i].sync_start_write();  // Invalidate cache before CPU write
      memcpy(ptr, input_data.data() + off, sz);
      in_bufs[i].sync_end_write();    // Flush cache after CPU write
      off += sz;
    }
  }

  // Buffers for capturing outputs
  std::vector<uint8_t> copy_out(out_bytes), zc_out(out_bytes);

  // -------------------------------------------------------------------------
  // Benchmark Loop
  // -------------------------------------------------------------------------

  Timer timer;
  Stats copy_in_stats, copy_invoke_stats, copy_out_stats, copy_total_stats;
  Stats zc_invoke_stats, zc_sync_stats, zc_total_stats;

  // Warmup runs to stabilize NPU performance
  for (int i = 0; i < warmup; i++) {
    copy_interp->Invoke();
    zc_interp->Invoke();
  }

  for (int iter = 0; iter < iterations; iter++) {
    // --- Copy-based path ---
    // This is the traditional approach: memcpy input, invoke, memcpy output
    double t_copy_in = 0, t_copy_invoke = 0, t_copy_out = 0;

    timer.start();
    {
      size_t off = 0;
      for (int idx : inputs) {
        size_t sz = copy_interp->tensor(idx)->bytes;
        memcpy(copy_interp->tensor(idx)->data.raw, input_data.data() + off, sz);
        off += sz;
      }
    }
    t_copy_in = timer.us();

    timer.start();
    copy_interp->Invoke();
    t_copy_invoke = timer.us();

    timer.start();
    {
      size_t off = 0;
      for (int idx : outputs) {
        size_t sz = copy_interp->tensor(idx)->bytes;
        memcpy(copy_out.data() + off, copy_interp->tensor(idx)->data.raw, sz);
        off += sz;
      }
    }
    t_copy_out = timer.us();

    copy_in_stats.add(t_copy_in);
    copy_invoke_stats.add(t_copy_invoke);
    copy_out_stats.add(t_copy_out);
    copy_total_stats.add(t_copy_in + t_copy_invoke + t_copy_out);

    // --- Zero-copy path ---
    // NPU reads directly from input dmabuf, writes directly to output dmabuf.
    // We only need to sync the output cache when CPU reads the results.
    double t_zc_invoke = 0, t_zc_sync = 0;

    timer.start();
    zc_interp->Invoke();
    t_zc_invoke = timer.us();

    // Sync output dmabuf for CPU read access
    timer.start();
    {
      size_t off = 0;
      for (size_t i = 0; i < outputs.size(); i++) {
        out_bufs[i].sync_start_read();  // Invalidate cache to see NPU's writes
        memcpy(zc_out.data() + off, out_bufs[i].map(), out_bufs[i].size());
        out_bufs[i].sync_end_read();
        off += out_bufs[i].size();
      }
    }
    t_zc_sync = timer.us();

    zc_invoke_stats.add(t_zc_invoke);
    zc_sync_stats.add(t_zc_sync);
    zc_total_stats.add(t_zc_invoke + t_zc_sync);
  }

  // -------------------------------------------------------------------------
  // Validate Results
  // -------------------------------------------------------------------------

  // Compare outputs - they should be identical (or very close for quantized)
  double maxd = max_diff(copy_out.data(), zc_out.data(), out_bytes);
  bool outputs_match = (maxd < 1.0);  // Allow tiny quantization differences

  std::cout << "--- Results ---\n";
  std::cout << "Output comparison: max diff = " << maxd;
  if (outputs_match) {
    std::cout << " [PASS]\n";
  } else {
    std::cout << " [FAIL - outputs differ significantly]\n";
  }

  // -------------------------------------------------------------------------
  // Report Results
  // -------------------------------------------------------------------------

  if (test_mode) {
    // Test mode: show detailed output comparison
    std::cout << "\nFirst 20 bytes of output:\n";
    std::cout << "  Copy:      ";
    for (int i = 0; i < std::min(20, (int)out_bytes); i++)
      std::cout << std::setw(3) << (int)copy_out[i] << " ";
    std::cout << "\n  Zero-copy: ";
    for (int i = 0; i < std::min(20, (int)out_bytes); i++)
      std::cout << std::setw(3) << (int)zc_out[i] << " ";
    std::cout << "\n";
  } else {
    // Benchmark mode: show timing statistics
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "\n--- Timing (µs) ---\n";
    std::cout << "                      mean      p50      p99\n";
    std::cout << "Copy-based:\n";
    std::cout << "  memcpy_in:      " << std::setw(8) << copy_in_stats.mean()
              << std::setw(9) << copy_in_stats.p50()
              << std::setw(9) << copy_in_stats.p99() << "\n";
    std::cout << "  invoke:         " << std::setw(8) << copy_invoke_stats.mean()
              << std::setw(9) << copy_invoke_stats.p50()
              << std::setw(9) << copy_invoke_stats.p99() << "\n";
    std::cout << "  memcpy_out:     " << std::setw(8) << copy_out_stats.mean()
              << std::setw(9) << copy_out_stats.p50()
              << std::setw(9) << copy_out_stats.p99() << "\n";
    std::cout << "  TOTAL:          " << std::setw(8) << copy_total_stats.mean()
              << std::setw(9) << copy_total_stats.p50()
              << std::setw(9) << copy_total_stats.p99() << "\n";
    std::cout << "Zero-copy:\n";
    std::cout << "  invoke:         " << std::setw(8) << zc_invoke_stats.mean()
              << std::setw(9) << zc_invoke_stats.p50()
              << std::setw(9) << zc_invoke_stats.p99() << "\n";
    std::cout << "  output_sync:    " << std::setw(8) << zc_sync_stats.mean()
              << std::setw(9) << zc_sync_stats.p50()
              << std::setw(9) << zc_sync_stats.p99() << "\n";
    std::cout << "  TOTAL:          " << std::setw(8) << zc_total_stats.mean()
              << std::setw(9) << zc_total_stats.p50()
              << std::setw(9) << zc_total_stats.p99() << "\n";

    double diff = copy_total_stats.mean() - zc_total_stats.mean();
    std::cout << "\nDifference: " << std::setprecision(0) << std::abs(diff) << " µs "
              << (diff > 0 ? "(zero-copy faster)" : "(copy faster)") << "\n";
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  vx::delegate::VxDelegateDelete(copy_del);
  vx::delegate::VxDelegateDelete(zc_del);

  return outputs_match ? 0 : 1;
}
