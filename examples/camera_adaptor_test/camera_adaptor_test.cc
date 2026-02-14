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

/**
 * CameraAdaptor Integration Test and Benchmark
 *
 * Tests RGBA→RGB conversion with DMA-BUF zero-copy pipeline.
 * Compares performance across different data path modes:
 *
 *   - copy-rgb:      G2D outputs RGB, memcpy to model input
 *   - copy-rgba:     G2D outputs RGBA, CPU drops alpha, memcpy to model
 *   - zerocopy-rgb:  G2D outputs RGB directly to DMA-BUF model input
 *   - zerocopy-rgba: G2D outputs RGBA to DMA-BUF, CameraAdaptor converts on NPU
 *
 * Test cases:
 * 1. MobileNet classification with grace_hopper.jpg
 * 2. YOLOv8 object detection with zidane.jpg
 *
 * Usage:
 *   ./camera_adaptor_test --model model.tflite --image input.jpg --mode zerocopy-rgba
 *   ./camera_adaptor_test --benchmark --model model.tflite --image input.jpg
 */

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "image_loader.h"
#include "g2d_helper.h"

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// VX Delegate
#include "delegate_main.h"
#include "vx_delegate_dmabuf.h"

// DMA-BUF support
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

using namespace camera_adaptor_test;

// ============================================================================
// Optimized UINT8 -> INT8 conversion helpers
// For INT8 quantized models with zero_point=-128, we need to convert
// camera UINT8 [0,255] to model INT8 [-128,127] by subtracting 128.
// ============================================================================

/**
 * Convert UINT8 RGB buffer to INT8 by subtracting 128.
 */
inline void ConvertUint8ToInt8(const uint8_t* src, int8_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<int8_t>(static_cast<int16_t>(src[i]) - 128);
  }
}

/**
 * Convert UINT8 RGBA buffer to INT8 RGB by dropping alpha and subtracting 128.
 */
inline void ConvertRgbaUint8ToRgbInt8(const uint8_t* src, int8_t* dst, int num_pixels) {
  for (int i = 0; i < num_pixels; ++i) {
    dst[i * 3 + 0] = static_cast<int8_t>(static_cast<int16_t>(src[i * 4 + 0]) - 128);
    dst[i * 3 + 1] = static_cast<int8_t>(static_cast<int16_t>(src[i * 4 + 1]) - 128);
    dst[i * 3 + 2] = static_cast<int8_t>(static_cast<int16_t>(src[i * 4 + 2]) - 128);
  }
}

/**
 * Convert UINT8 RGBA buffer to UINT8 RGB by dropping alpha.
 */
inline void ConvertRgbaUint8ToRgbUint8(const uint8_t* src, uint8_t* dst, int num_pixels) {
  for (int i = 0; i < num_pixels; ++i) {
    dst[i * 3 + 0] = src[i * 4 + 0];
    dst[i * 3 + 1] = src[i * 4 + 1];
    dst[i * 3 + 2] = src[i * 4 + 2];
  }
}

// DMA heap type
enum class HeapType { Auto, Cached, Uncached };

const char* HeapTypeToString(HeapType heap) {
  switch (heap) {
    case HeapType::Auto: return "auto";
    case HeapType::Cached: return "cached";
    case HeapType::Uncached: return "uncached";
    default: return "unknown";
  }
}

// Data path modes
enum class DataPathMode {
  CopyRgb,       // G2D -> RGB buffer -> memcpy to model
  CopyRgba,      // G2D -> RGBA buffer -> CPU alpha drop -> memcpy to model
  ZeroCopyRgb,   // G2D -> DMA-BUF (RGB) -> direct model input
  ZeroCopyRgba,  // G2D -> DMA-BUF (RGBA) -> CameraAdaptor -> model input
};

const char* DataPathModeToString(DataPathMode mode) {
  switch (mode) {
    case DataPathMode::CopyRgb: return "copy-rgb";
    case DataPathMode::CopyRgba: return "copy-rgba";
    case DataPathMode::ZeroCopyRgb: return "zerocopy-rgb";
    case DataPathMode::ZeroCopyRgba: return "zerocopy-rgba";
    default: return "unknown";
  }
}

DataPathMode DataPathModeFromString(const std::string& str) {
  if (str == "copy-rgb") return DataPathMode::CopyRgb;
  if (str == "copy-rgba") return DataPathMode::CopyRgba;
  if (str == "zerocopy-rgb") return DataPathMode::ZeroCopyRgb;
  if (str == "zerocopy-rgba") return DataPathMode::ZeroCopyRgba;
  // Default
  return DataPathMode::ZeroCopyRgba;
}

// Command-line configuration
struct Config {
  std::string model_path;
  std::string image_path;
  std::string labels_path;
  DataPathMode mode = DataPathMode::ZeroCopyRgba;
  std::string task = "classification";  // or "detection"
  std::string expected_class;
  std::string expected_detections;
  HeapType heap = HeapType::Auto;
  int iterations = 20;
  int warmup = 5;
  int top_k = 5;
  float confidence_threshold = 0.25f;
  float nms_threshold = 0.45f;
  bool benchmark = false;
  bool letterbox = true;
  bool verbose = false;
  bool use_g2d = true;  // Use G2D hardware resize when available
};

// Classification result
struct ClassificationResult {
  int class_id;
  std::string label;
  float score;
};

// Detection result (for YOLO)
struct Detection {
  int class_id;
  std::string label;
  float confidence;
  float x1, y1, x2, y2;  // Bounding box in pixel coordinates
};

// Benchmark result for a single mode
struct BenchmarkResult {
  DataPathMode mode;
  std::string model_name;
  double mean_ms;
  double min_ms;
  double max_ms;
  double std_ms;
  // Timing breakdown (all per-frame averages)
  double resize_mean_ms;  // G2D resize per frame
  double copy_mean_ms;    // Cache invalidate + mmap + copy (copy mode)
                          // or G2D finish only (zerocopy mode)
  double invoke_mean_ms;  // NPU invoke per frame
  bool results_valid;
};

// Print usage
void PrintUsage(const char* program) {
  std::cout << "CameraAdaptor Integration Test and Benchmark\n\n"
            << "Usage: " << program << " [options]\n\n"
            << "Options:\n"
            << "  --model PATH        Path to TFLite model (required)\n"
            << "  --image PATH        Path to input image (required)\n"
            << "  --labels PATH       Path to labels file\n"
            << "  --mode MODE         Data path mode (default: zerocopy-rgba)\n"
            << "                        copy-rgb:      G2D->RGB->memcpy\n"
            << "                        copy-rgba:     G2D->RGBA->CPU drop A->memcpy\n"
            << "                        zerocopy-rgb:  G2D->DMA-BUF RGB->direct\n"
            << "                        zerocopy-rgba: G2D->DMA-BUF RGBA->CameraAdaptor\n"
            << "  --task TYPE         Task type: classification, detection (default: classification)\n"
            << "  --expected_class    Expected class name for classification\n"
            << "  --expected_detections  Expected detections: \"person:2,tie:1\"\n"
            << "  --heap TYPE         DMA heap type: auto, cached, uncached (default: auto)\n"
            << "  --iterations N      Number of inference iterations (default: 20)\n"
            << "  --warmup N          Number of warmup iterations (default: 5)\n"
            << "  --top_k N           Show top-K results for classification (default: 5)\n"
            << "  --threshold F       Confidence threshold for detection (default: 0.25)\n"
            << "  --nms F             NMS threshold for detection (default: 0.45)\n"
            << "  --benchmark         Run all modes and compare\n"
            << "  --no-letterbox      Disable letterbox (stretch to fit)\n"
            << "  --no-g2d            Use software resize instead of G2D\n"
            << "  --verbose           Enable verbose logging\n"
            << "  --help              Show this help\n\n"
            << "Examples:\n"
            << "  # Single mode test\n"
            << "  " << program << " --model mobilenet.tflite --image grace_hopper.jpg \\\n"
            << "      --labels labels.txt --mode zerocopy-rgba\n\n"
            << "  # Benchmark all modes\n"
            << "  " << program << " --benchmark --model mobilenet.tflite --image grace_hopper.jpg\n\n"
            << "  # YOLOv8 detection\n"
            << "  " << program << " --task detection --model yolov8n.tflite --image zidane.jpg \\\n"
            << "      --labels coco.txt --expected_detections \"person:2,tie:1\"\n";
}

// Parse command-line arguments
Config ParseArgs(int argc, char* argv[]) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--model" && i + 1 < argc) {
      config.model_path = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      config.image_path = argv[++i];
    } else if (arg == "--labels" && i + 1 < argc) {
      config.labels_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      config.mode = DataPathModeFromString(argv[++i]);
    } else if (arg == "--task" && i + 1 < argc) {
      config.task = argv[++i];
    } else if (arg == "--expected_class" && i + 1 < argc) {
      config.expected_class = argv[++i];
    } else if (arg == "--expected_detections" && i + 1 < argc) {
      config.expected_detections = argv[++i];
    } else if (arg == "--heap" && i + 1 < argc) {
      std::string val = argv[++i];
      if (val == "cached") config.heap = HeapType::Cached;
      else if (val == "uncached") config.heap = HeapType::Uncached;
      else config.heap = HeapType::Auto;
    } else if (arg == "--iterations" && i + 1 < argc) {
      config.iterations = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      config.warmup = std::stoi(argv[++i]);
    } else if (arg == "--top_k" && i + 1 < argc) {
      config.top_k = std::stoi(argv[++i]);
    } else if (arg == "--threshold" && i + 1 < argc) {
      config.confidence_threshold = std::stof(argv[++i]);
    } else if (arg == "--nms" && i + 1 < argc) {
      config.nms_threshold = std::stof(argv[++i]);
    } else if (arg == "--benchmark") {
      config.benchmark = true;
    } else if (arg == "--no-letterbox") {
      config.letterbox = false;
    } else if (arg == "--no-g2d") {
      config.use_g2d = false;
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      exit(0);
    }
  }

  return config;
}

// Probe available DMA heaps and print results
void ProbeDmaHeaps() {
  const char* heaps[] = {
      "/dev/dma_heap/linux,cma",
      "/dev/dma_heap/linux,cma-uncached",
      "/dev/dma_heap/reserved-uncached",
      "/dev/dma_heap/system",
      nullptr
  };

  std::cout << "DMA heaps:\n";
  for (int i = 0; heaps[i]; ++i) {
    int fd = open(heaps[i], O_RDWR);
    if (fd >= 0) {
      std::cout << "  " << heaps[i] << ": available\n";
      close(fd);
    } else {
      std::cout << "  " << heaps[i] << ": not available\n";
    }
  }
}

// Allocate DMA-BUF from dma_heap with heap type selection
int AllocateDmaBuf(size_t size, void** map_ptr, HeapType heap = HeapType::Auto) {
  const char* cached_heaps[] = {
      "/dev/dma_heap/linux,cma",
      "/dev/dma_heap/system",
      nullptr
  };
  const char* uncached_heaps[] = {
      "/dev/dma_heap/linux,cma-uncached",
      "/dev/dma_heap/reserved-uncached",
      nullptr
  };

  const char* heap_used = nullptr;
  int heap_fd = -1;

  auto try_heaps = [&](const char** paths) -> bool {
    for (int i = 0; paths[i]; ++i) {
      heap_fd = open(paths[i], O_RDWR);
      if (heap_fd >= 0) {
        heap_used = paths[i];
        return true;
      }
    }
    return false;
  };

  switch (heap) {
    case HeapType::Cached:
      try_heaps(cached_heaps);
      break;
    case HeapType::Uncached:
      try_heaps(uncached_heaps);
      break;
    case HeapType::Auto:
      // Try uncached first (optimal for H2H pipeline), fall back to cached
      if (!try_heaps(uncached_heaps)) {
        try_heaps(cached_heaps);
      }
      break;
  }

  if (heap_fd < 0) {
    std::cerr << "Failed to open dma_heap device (heap=" << HeapTypeToString(heap)
              << ")" << std::endl;
    return -1;
  }

  std::cout << "DMA-BUF heap: " << heap_used << "\n";

  struct dma_heap_allocation_data alloc_data = {};
  alloc_data.len = size;
  alloc_data.fd_flags = O_RDWR | O_CLOEXEC;

  if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
    std::cerr << "DMA_HEAP_IOCTL_ALLOC failed: " << strerror(errno) << std::endl;
    close(heap_fd);
    return -1;
  }

  close(heap_fd);

  if (map_ptr) {
    *map_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    alloc_data.fd, 0);
    if (*map_ptr == MAP_FAILED) {
      std::cerr << "mmap failed: " << strerror(errno) << std::endl;
      close(alloc_data.fd);
      return -1;
    }
  }

  return alloc_data.fd;
}

void FreeDmaBuf(int fd, void* map_ptr, size_t size) {
  if (map_ptr && map_ptr != MAP_FAILED) {
    munmap(map_ptr, size);
  }
  if (fd >= 0) {
    close(fd);
  }
}

void SyncDmaBufForCpu(int fd, bool for_write) {
  struct dma_buf_sync sync = {};
  sync.flags = DMA_BUF_SYNC_START;
  sync.flags |= for_write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;
  ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

void SyncDmaBufForDevice(int fd) {
  struct dma_buf_sync sync = {};
  sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
  ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

// Get top-K classification results
std::vector<ClassificationResult> GetTopK(
    const float* scores, int num_classes,
    const std::vector<std::string>& labels, int k) {

  std::vector<std::pair<float, int>> score_index;
  for (int i = 0; i < num_classes; ++i) {
    score_index.emplace_back(scores[i], i);
  }

  std::partial_sort(score_index.begin(), score_index.begin() + k,
                    score_index.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<ClassificationResult> results;
  for (int i = 0; i < k && i < static_cast<int>(score_index.size()); ++i) {
    ClassificationResult r;
    r.class_id = score_index[i].second;
    r.score = score_index[i].first;
    r.label = (r.class_id < static_cast<int>(labels.size()))
              ? labels[r.class_id] : "unknown";
    results.push_back(r);
  }

  return results;
}

std::vector<ClassificationResult> GetTopKQuantized(
    const uint8_t* output, int num_classes,
    const std::vector<std::string>& labels, int k,
    float scale, int zero_point) {

  std::vector<float> scores(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    scores[i] = (output[i] - zero_point) * scale;
  }

  return GetTopK(scores.data(), num_classes, labels, k);
}

// Compute IoU for NMS
float ComputeIoU(const Detection& a, const Detection& b) {
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);

  float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
  float b_area = (b.x2 - b.x1) * (b.y2 - b.y1);

  return inter_area / (a_area + b_area - inter_area + 1e-6f);
}

// Non-Maximum Suppression
std::vector<Detection> ApplyNMS(std::vector<Detection>& detections, float threshold) {
  // Sort by confidence
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> result;
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) continue;

    result.push_back(detections[i]);

    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j]) continue;
      if (detections[i].class_id != detections[j].class_id) continue;

      if (ComputeIoU(detections[i], detections[j]) > threshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

// Dequantize INT8 value to float
inline float Dequantize(int8_t value, float scale, int32_t zero_point) {
  return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// Parse YOLOv8 detection output from INT8 quantized tensor
std::vector<Detection> ParseYoloV8OutputInt8(
    const int8_t* output, int num_detections, int num_classes,
    int img_width, int img_height,
    const std::vector<std::string>& labels,
    float conf_threshold, float scale, int32_t zero_point) {

  std::vector<Detection> detections;

  // YOLOv8 output is transposed: [4 + num_classes, num_detections]
  const int stride = num_detections;

  // Model input size for coordinate scaling (YOLOv8 outputs normalized coords)
  // The img_width/img_height here are model input dimensions, not original image
  float scale_x = static_cast<float>(img_width);
  float scale_y = static_cast<float>(img_height);

  for (int i = 0; i < num_detections; ++i) {
    // Get box coordinates (first 4 values) - dequantize and scale to pixels
    float x_center = Dequantize(output[0 * stride + i], scale, zero_point) * scale_x;
    float y_center = Dequantize(output[1 * stride + i], scale, zero_point) * scale_y;
    float w = Dequantize(output[2 * stride + i], scale, zero_point) * scale_x;
    float h = Dequantize(output[3 * stride + i], scale, zero_point) * scale_y;

    // Find best class
    int best_class = -1;
    float best_score = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      float score = Dequantize(output[(4 + c) * stride + i], scale, zero_point);
      if (score > best_score) {
        best_score = score;
        best_class = c;
      }
    }

    if (best_score < conf_threshold) continue;

    Detection det;
    det.class_id = best_class;
    det.confidence = best_score;
    det.label = (best_class < static_cast<int>(labels.size()))
                ? labels[best_class] : "class_" + std::to_string(best_class);

    // Convert to corner format
    det.x1 = (x_center - w / 2.0f);
    det.y1 = (y_center - h / 2.0f);
    det.x2 = (x_center + w / 2.0f);
    det.y2 = (y_center + h / 2.0f);

    // Clamp to image bounds
    det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(img_width)));
    det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(img_height)));
    det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(img_width)));
    det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(img_height)));

    detections.push_back(det);
  }

  return detections;
}

// Parse YOLOv8 detection output
// YOLOv8 output format: [batch, 84, 8400] for 80 COCO classes
// Each of the 8400 predictions: [x_center, y_center, width, height, class_scores...]
std::vector<Detection> ParseYoloV8Output(
    const float* output, int num_detections, int num_classes,
    int img_width, int img_height,
    const std::vector<std::string>& labels,
    float conf_threshold) {

  std::vector<Detection> detections;

  // YOLOv8 output is transposed: [4 + num_classes, num_detections]
  const int stride = num_detections;

  for (int i = 0; i < num_detections; ++i) {
    // Get box coordinates (first 4 values)
    float x_center = output[0 * stride + i];
    float y_center = output[1 * stride + i];
    float w = output[2 * stride + i];
    float h = output[3 * stride + i];

    // Find best class
    int best_class = -1;
    float best_score = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      float score = output[(4 + c) * stride + i];
      if (score > best_score) {
        best_score = score;
        best_class = c;
      }
    }

    if (best_score < conf_threshold) continue;

    Detection det;
    det.class_id = best_class;
    det.confidence = best_score;
    det.label = (best_class < static_cast<int>(labels.size()))
                ? labels[best_class] : "class_" + std::to_string(best_class);

    // Convert to corner format and scale to image size
    det.x1 = (x_center - w / 2.0f);
    det.y1 = (y_center - h / 2.0f);
    det.x2 = (x_center + w / 2.0f);
    det.y2 = (y_center + h / 2.0f);

    // Clamp to image bounds
    det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(img_width)));
    det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(img_height)));
    det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(img_width)));
    det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(img_height)));

    detections.push_back(det);
  }

  return detections;
}

// Parse expected detections string ("person:2,tie:1")
std::map<std::string, int> ParseExpectedDetections(const std::string& str) {
  std::map<std::string, int> expected;
  if (str.empty()) return expected;

  size_t pos = 0;
  std::string token;
  std::string s = str;

  while ((pos = s.find(',')) != std::string::npos) {
    token = s.substr(0, pos);
    size_t colon = token.find(':');
    if (colon != std::string::npos) {
      expected[token.substr(0, colon)] = std::stoi(token.substr(colon + 1));
    }
    s.erase(0, pos + 1);
  }
  // Last token
  size_t colon = s.find(':');
  if (colon != std::string::npos) {
    expected[s.substr(0, colon)] = std::stoi(s.substr(colon + 1));
  }

  return expected;
}

// Run classification test
bool RunClassificationTest(const Config& config,
                           tflite::Interpreter* interpreter,
                           const std::vector<std::string>& labels) {
  std::cout << "\n=== Classification Results ===" << std::endl;

  TfLiteTensor* output_tensor = interpreter->output_tensor(0);
  int num_classes = output_tensor->dims->data[output_tensor->dims->size - 1];

  std::vector<ClassificationResult> results;

  if (output_tensor->type == kTfLiteInt8) {
    auto params = output_tensor->params;
    // Reinterpret INT8 data as signed for proper dequantization
    const int8_t* data = interpreter->typed_output_tensor<int8_t>(0);
    std::vector<uint8_t> unsigned_data(num_classes);
    for (int i = 0; i < num_classes; ++i) {
      // Convert INT8 [-128,127] back to UINT8 [0,255] for GetTopKQuantized
      unsigned_data[i] = static_cast<uint8_t>(static_cast<int>(data[i]) + 128);
    }
    results = GetTopKQuantized(
        unsigned_data.data(),
        num_classes, labels, config.top_k,
        params.scale, params.zero_point + 128);  // Adjust zero_point too
  } else if (output_tensor->type == kTfLiteUInt8) {
    auto params = output_tensor->params;
    results = GetTopKQuantized(
        interpreter->typed_output_tensor<uint8_t>(0),
        num_classes, labels, config.top_k,
        params.scale, params.zero_point);
  } else if (output_tensor->type == kTfLiteFloat32) {
    results = GetTopK(
        interpreter->typed_output_tensor<float>(0),
        num_classes, labels, config.top_k);
  } else {
    std::cerr << "Unsupported output type: " << output_tensor->type << std::endl;
    return false;
  }

  std::cout << "\nTop-" << config.top_k << " predictions:\n";
  for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "  " << (i + 1) << ". [" << results[i].class_id << "] "
              << results[i].label << ": "
              << std::fixed << std::setprecision(4) << results[i].score
              << std::endl;
  }

  if (!config.expected_class.empty()) {
    bool found = false;
    for (const auto& r : results) {
      if (r.label.find(config.expected_class) != std::string::npos) {
        found = true;
        std::cout << "\n[PASS] Expected class \"" << config.expected_class
                  << "\" found in top-" << config.top_k << std::endl;
        break;
      }
    }
    if (!found) {
      std::cout << "\n[FAIL] Expected class \"" << config.expected_class
                << "\" NOT found in top-" << config.top_k << std::endl;
      return false;
    }
  }

  return true;
}

// Run detection test with full YOLOv8 box decoding and NMS
bool RunDetectionTest(const Config& config,
                      tflite::Interpreter* interpreter,
                      const std::vector<std::string>& labels,
                      int img_width, int img_height) {
  std::cout << "\n=== Detection Results ===" << std::endl;

  TfLiteTensor* output_tensor = interpreter->output_tensor(0);

  // Print output shape
  std::cout << "Output shape: [";
  for (int i = 0; i < output_tensor->dims->size; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << output_tensor->dims->data[i];
  }
  std::cout << "]\n";

  // YOLOv8 output: [1, 84, num_detections] or [1, num_detections, 84]
  // 84 = 4 (box coords) + 80 (COCO classes)
  int num_classes = 80;  // COCO classes
  int num_detections = 0;
  bool transposed = false;

  if (output_tensor->dims->size == 3) {
    int dim1 = output_tensor->dims->data[1];
    int dim2 = output_tensor->dims->data[2];
    if (dim1 == 84 || dim1 == 4 + num_classes) {
      // [1, 84, num_detections] - transposed format
      num_detections = dim2;
      transposed = true;
    } else if (dim2 == 84 || dim2 == 4 + num_classes) {
      // [1, num_detections, 84] - standard format
      num_detections = dim1;
      transposed = false;
    }
  }

  if (num_detections == 0) {
    std::cerr << "Unrecognized YOLOv8 output shape" << std::endl;
    return false;
  }

  std::cout << "Detections: " << num_detections << " candidates (transposed="
            << (transposed ? "yes" : "no") << ")\n";

  std::vector<Detection> detections;

  if (output_tensor->type == kTfLiteInt8) {
    const int8_t* data = interpreter->typed_output_tensor<int8_t>(0);
    if (!data) {
      std::cerr << "Failed to get INT8 output data" << std::endl;
      return false;
    }
    auto params = output_tensor->params;
    std::cout << "Type: INT8 (scale=" << std::scientific << params.scale
              << ", zp=" << params.zero_point << ")\n" << std::fixed;

    detections = ParseYoloV8OutputInt8(
        data, num_detections, num_classes,
        img_width, img_height, labels,
        config.confidence_threshold, params.scale, params.zero_point);

  } else if (output_tensor->type == kTfLiteFloat32) {
    const float* data = interpreter->typed_output_tensor<float>(0);
    if (!data) {
      std::cerr << "Failed to get FLOAT32 output data" << std::endl;
      return false;
    }
    std::cout << "Type: FLOAT32\n";

    detections = ParseYoloV8Output(
        data, num_detections, num_classes,
        img_width, img_height, labels,
        config.confidence_threshold);

  } else {
    std::cerr << "Unsupported output type: " << output_tensor->type << std::endl;
    return false;
  }

  std::cout << "Pre-NMS detections: " << detections.size() << "\n";

  // Apply NMS
  auto final_detections = ApplyNMS(detections, config.nms_threshold);

  std::cout << "Post-NMS detections: " << final_detections.size() << "\n\n";

  // Print detections
  if (final_detections.empty()) {
    std::cout << "No objects detected.\n";
  } else {
    std::cout << "Detected objects:\n";
    std::map<std::string, int> class_counts;
    for (const auto& det : final_detections) {
      std::cout << "  - " << det.label << " (" << std::fixed
                << std::setprecision(1) << (det.confidence * 100.0f) << "%)"
                << " [" << static_cast<int>(det.x1) << ","
                << static_cast<int>(det.y1) << ","
                << static_cast<int>(det.x2) << ","
                << static_cast<int>(det.y2) << "]\n";
      class_counts[det.label]++;
    }

    // Print summary
    std::cout << "\nSummary: ";
    bool first = true;
    for (const auto& [label, count] : class_counts) {
      if (!first) std::cout << ", ";
      std::cout << count << " " << label;
      if (count > 1) std::cout << "s";
      first = false;
    }
    std::cout << "\n";
  }

  // Validate expected detections if specified
  if (!config.expected_detections.empty()) {
    auto expected = ParseExpectedDetections(config.expected_detections);

    // Count actual detections by class
    std::map<std::string, int> actual_counts;
    for (const auto& det : final_detections) {
      actual_counts[det.label]++;
    }

    bool all_match = true;
    std::cout << "\nValidation:\n";
    for (const auto& [label, expected_count] : expected) {
      int actual_count = actual_counts[label];
      bool match = (actual_count >= expected_count);
      std::cout << "  " << label << ": expected " << expected_count
                << ", got " << actual_count
                << (match ? " [PASS]" : " [FAIL]") << "\n";
      if (!match) all_match = false;
    }

    if (all_match) {
      std::cout << "\n[PASS] All expected detections found\n";
      return true;
    } else {
      std::cout << "\n[FAIL] Missing expected detections\n";
      return false;
    }
  }

  return true;
}

// Helper to extract model basename
std::string GetModelName(const std::string& path) {
  size_t last_slash = path.find_last_of("/\\");
  std::string filename = (last_slash != std::string::npos)
                         ? path.substr(last_slash + 1) : path;
  size_t ext = filename.find_last_of('.');
  return (ext != std::string::npos) ? filename.substr(0, ext) : filename;
}

// Run a single inference pass with timing
struct InferenceRun {
  double resize_ms;
  double copy_ms;
  double invoke_ms;
  double total_ms;
  bool success;
};

InferenceRun RunSingleInference(
    const Config& config,
    DataPathMode mode,
    tflite::Interpreter* interpreter,
    TfLiteDelegate* delegate,
    const Image& source_image,
    int input_width, int input_height, int input_channels,
    TfLiteBufferHandle buffer_handle,
    int dmabuf_fd, void* dmabuf_ptr, size_t dmabuf_size,
    G2DBuffer* g2d_output) {

  InferenceRun run = {};
  auto total_start = std::chrono::high_resolution_clock::now();

  bool use_rgba = (mode == DataPathMode::CopyRgba || mode == DataPathMode::ZeroCopyRgba);
  bool use_zerocopy = (mode == DataPathMode::ZeroCopyRgb || mode == DataPathMode::ZeroCopyRgba);

  // --- Resize phase ---
  auto resize_start = std::chrono::high_resolution_clock::now();

  Image resized;
  if (config.use_g2d && G2DIsAvailable() && g2d_output && g2d_output->virt_addr) {
    // G2D resize already done into g2d_output buffer
    // Just need to flush cache if we're using CPU access after
    if (!use_zerocopy) {
      // Will access from CPU - data is already in G2D buffer
    }
  } else {
    // Software resize fallback
    resized = ResizeImage(source_image, input_width, input_height, config.letterbox, 0x808080);
    if (use_rgba) {
      resized = ConvertToRGBA(resized);
    }
  }

  auto resize_end = std::chrono::high_resolution_clock::now();
  run.resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

  // --- Copy/setup phase ---
  auto copy_start = std::chrono::high_resolution_clock::now();

  if (use_zerocopy && buffer_handle != kTfLiteNullBufferHandle) {
    // Zero-copy: data is already in DMA-BUF from G2D
    // Just sync for device
    SyncDmaBufForDevice(dmabuf_fd);
  } else {
    // Copy mode: copy data to input tensor
    TfLiteTensor* input_tensor = interpreter->input_tensor(0);

    const uint8_t* src_data = nullptr;
    size_t src_size = 0;

    if (g2d_output && g2d_output->virt_addr) {
      src_data = static_cast<const uint8_t*>(g2d_output->virt_addr);
      src_size = g2d_output->size;
    } else if (resized.IsValid()) {
      src_data = resized.data.data();
      src_size = resized.GetByteSize();
    }

    if (src_data && src_size > 0) {
      if (use_rgba && !use_zerocopy) {
        // Need to drop alpha channel on CPU
        // RGBA -> RGB conversion (with INT8 shift if needed)
        int num_pixels = input_width * input_height;
        if (input_tensor->type == kTfLiteInt8) {
          // INT8 model: UINT8 RGBA → drop alpha + subtract 128 → INT8 RGB
          int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
          ConvertRgbaUint8ToRgbInt8(src_data, dst, num_pixels);
        } else {
          // UINT8 model: just drop alpha
          uint8_t* dst = input_tensor->data.uint8;
          for (int i = 0; i < num_pixels; ++i) {
            dst[i * 3 + 0] = src_data[i * 4 + 0];
            dst[i * 3 + 1] = src_data[i * 4 + 1];
            dst[i * 3 + 2] = src_data[i * 4 + 2];
          }
        }
      } else if (input_tensor->type == kTfLiteInt8) {
        // INT8 model with RGB input: UINT8 RGB → subtract 128 → INT8 RGB
        int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
        size_t count = std::min(src_size, static_cast<size_t>(input_tensor->bytes));
        ConvertUint8ToInt8(src_data, dst, count);
      } else if (input_tensor->type == kTfLiteUInt8) {
        std::memcpy(input_tensor->data.uint8, src_data,
                    std::min(src_size, static_cast<size_t>(input_tensor->bytes)));
      } else if (input_tensor->type == kTfLiteFloat32) {
        float* dst = interpreter->typed_input_tensor<float>(0);
        size_t count = std::min(src_size, static_cast<size_t>(input_tensor->bytes / sizeof(float)));
        for (size_t i = 0; i < count; ++i) {
          dst[i] = src_data[i] / 255.0f;
        }
      }
    }
  }

  auto copy_end = std::chrono::high_resolution_clock::now();
  run.copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();

  // --- Invoke phase ---
  auto invoke_start = std::chrono::high_resolution_clock::now();

  run.success = (interpreter->Invoke() == kTfLiteOk);

  auto invoke_end = std::chrono::high_resolution_clock::now();
  run.invoke_ms = std::chrono::duration<double, std::milli>(invoke_end - invoke_start).count();

  auto total_end = std::chrono::high_resolution_clock::now();
  run.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

  return run;
}

// Run test for a specific mode
BenchmarkResult RunModeTest(const Config& config, DataPathMode mode,
                            const std::vector<std::string>& labels) {
  BenchmarkResult result = {};  // Zero-initialize all fields
  result.mode = mode;
  result.model_name = GetModelName(config.model_path);
  result.results_valid = false;

  bool use_rgba = (mode == DataPathMode::CopyRgba || mode == DataPathMode::ZeroCopyRgba);
  bool use_zerocopy = (mode == DataPathMode::ZeroCopyRgb || mode == DataPathMode::ZeroCopyRgba);
  // Use adaptor for any zerocopy mode (RGBA needs Slice, and INT8 models need Sub)
  bool use_adaptor = use_zerocopy;

  std::cout << "\n========================================\n"
            << "Mode: " << DataPathModeToString(mode) << "\n"
            << "  RGBA output: " << (use_rgba ? "yes" : "no") << "\n"
            << "  Zero-copy:   " << (use_zerocopy ? "yes" : "no") << "\n"
            << "  Adaptor:     " << (use_adaptor ? "yes" : "no") << "\n"
            << "========================================\n";

  // Load model
  auto model = tflite::FlatBufferModel::BuildFromFile(config.model_path.c_str());
  if (!model) {
    std::cerr << "Failed to load model: " << config.model_path << std::endl;
    return result;
  }

  // Create interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    std::cerr << "Failed to create interpreter" << std::endl;
    return result;
  }

  // Get input tensor info
  TfLiteTensor* input_tensor_info = interpreter->input_tensor(0);
  int input_height = input_tensor_info->dims->data[1];
  int input_width = input_tensor_info->dims->data[2];
  int input_channels = input_tensor_info->dims->data[3];

  std::cout << "Model input: " << input_width << "x" << input_height
            << "x" << input_channels << std::endl;

  // Create delegate
  vx::delegate::VxDelegateOptions options = vx::delegate::VxDelegateOptionsDefault();
  options.enable_dmabuf = use_zerocopy;
  TfLiteDelegate* delegate = vx::delegate::VxDelegateCreate(&options);
  if (!delegate) {
    std::cerr << "Failed to create VX delegate" << std::endl;
    return result;
  }

  // Load image
  Image img = LoadImage(config.image_path, 3);
  if (!img.IsValid()) {
    std::cerr << "Failed to load image: " << config.image_path << std::endl;
    vx::delegate::VxDelegateDelete(delegate);
    return result;
  }
  std::cout << "Loaded image: " << img.width << "x" << img.height << "\n";

  // Determine buffer format and size
  int buffer_channels = use_rgba ? 4 : 3;
  size_t buffer_size = static_cast<size_t>(input_width) * input_height * buffer_channels;

  // Allocate G2D buffer for resize output (if using G2D)
  G2DBuffer g2d_output = {};
  bool g2d_available = config.use_g2d && G2DIsAvailable();

  if (g2d_available) {
#ifdef HAVE_G2D
    uint32_t g2d_format = use_rgba ? G2D_RGBA8888 : G2D_RGB888;
    bool g2d_cacheable = (config.heap != HeapType::Uncached);
    if (!G2DAllocBuffer(&g2d_output, input_width, input_height, g2d_format, g2d_cacheable)) {
      std::cerr << "G2D buffer allocation failed, using software resize\n";
      g2d_available = false;
    } else {
      std::cout << "Allocated G2D output buffer: " << buffer_size << " bytes"
                << (g2d_cacheable ? " (cached)" : " (uncached)") << "\n";
    }
#else
    g2d_available = false;
#endif
  }

  // DMA-BUF setup for zero-copy modes
  int dmabuf_fd = -1;
  void* dmabuf_ptr = nullptr;
  TfLiteBufferHandle buffer_handle = kTfLiteNullBufferHandle;

  if (use_zerocopy) {
    dmabuf_fd = AllocateDmaBuf(buffer_size, &dmabuf_ptr, config.heap);
    if (dmabuf_fd < 0) {
      std::cerr << "Failed to allocate DMA-BUF" << std::endl;
      G2DFreeBuffer(&g2d_output);
      vx::delegate::VxDelegateDelete(delegate);
      return result;
    }
    std::cout << "Allocated DMA-BUF: fd=" << dmabuf_fd << ", size=" << buffer_size << "\n";

    buffer_handle = VxDelegateRegisterDmaBuf(delegate, dmabuf_fd, buffer_size,
                                              kVxDmaBufSyncNone);
    if (buffer_handle == kTfLiteNullBufferHandle) {
      std::cerr << "Failed to register DMA-BUF" << std::endl;
      FreeDmaBuf(dmabuf_fd, dmabuf_ptr, buffer_size);
      G2DFreeBuffer(&g2d_output);
      vx::delegate::VxDelegateDelete(delegate);
      return result;
    }

    int input_tensor_idx = interpreter->inputs()[0];
    if (VxDelegateBindDmaBufToTensor(delegate, buffer_handle, input_tensor_idx) != kTfLiteOk) {
      std::cerr << "Failed to bind DMA-BUF to tensor" << std::endl;
    } else {
      std::cout << "Bound DMA-BUF to input tensor\n";
    }
  }

  // Configure CameraAdaptor for zero-copy modes
  // RGBA needs Slice+Reverse, RGB with INT8 models needs Sub for UINT8→INT8
  if (use_adaptor) {
    int input_tensor_idx = interpreter->inputs()[0];
    const char* format = use_rgba ? "rgba" : "rgb";
    if (VxCameraAdaptorSetFormat(delegate, input_tensor_idx, format) == kTfLiteOk) {
      std::cout << "Configured CameraAdaptor: " << format << " -> rgb\n";
    } else {
      std::cerr << "Failed to configure CameraAdaptor\n";
    }
  }

  // Apply delegate
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
    std::cerr << "Failed to apply delegate" << std::endl;
    if (buffer_handle != kTfLiteNullBufferHandle) {
      VxDelegateUnregisterDmaBuf(delegate, buffer_handle);
    }
    FreeDmaBuf(dmabuf_fd, dmabuf_ptr, buffer_size);
    G2DFreeBuffer(&g2d_output);
    vx::delegate::VxDelegateDelete(delegate);
    return result;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors" << std::endl;
    if (buffer_handle != kTfLiteNullBufferHandle) {
      VxDelegateUnregisterDmaBuf(delegate, buffer_handle);
    }
    FreeDmaBuf(dmabuf_fd, dmabuf_ptr, buffer_size);
    G2DFreeBuffer(&g2d_output);
    vx::delegate::VxDelegateDelete(delegate);
    return result;
  }

  // Pre-process image once with G2D (or software fallback)
  // Time the resize operation
  double preprocess_resize_ms = 0.0;
  double preprocess_copy_to_dmabuf_ms = 0.0;

  Image processed_image;

  // For zerocopy modes, try G2DResizeToDmaBuf first (true zero-copy if available)
  bool used_zerocopy_resize = false;
  if (use_zerocopy && g2d_available && dmabuf_ptr) {
#ifdef HAVE_G2D
    // Convert source to RGBA for G2D input
    Image rgba_src = ConvertToRGBA(img);

    uint32_t dst_format = use_rgba ? G2D_RGBA8888 : G2D_RGB888;

    // Try G2D resize directly to DMA-BUF (true zero-copy)
    auto resize_start = std::chrono::high_resolution_clock::now();

    if (G2DResizeToDmaBuf(rgba_src.data.data(), rgba_src.width, rgba_src.height,
                          rgba_src.channels, dmabuf_fd, dmabuf_ptr,
                          input_width, input_height, dst_format,
                          config.letterbox, 0x808080)) {
      auto resize_end = std::chrono::high_resolution_clock::now();
      double g2d_only_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

      // Sync DMA-BUF for device access after G2D write
      // Note: For G2D→NPU (both using physical addresses), this is a no-op
      // (~3µs) but included for correctness. The kernel may need this hint
      // for cache maintenance on some platforms.
      auto sync_start = std::chrono::high_resolution_clock::now();
      SyncDmaBufForDevice(dmabuf_fd);
      auto sync_end = std::chrono::high_resolution_clock::now();
      double sync_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

      preprocess_resize_ms = g2d_only_ms + sync_ms;
      std::cout << "G2D resize → DMA-BUF: " << std::fixed << std::setprecision(2)
                << g2d_only_ms << " ms (sync: " << std::setprecision(3) << sync_ms << " ms)\n";
      used_zerocopy_resize = true;
    }
    // If G2DResizeToDmaBuf fails, fall through to two-buffer path
#endif
  }

  // Fallback: G2D resize to intermediate buffer, then copy to DMA-BUF or tensor
  if (!used_zerocopy_resize) {
    if (g2d_available && g2d_output.virt_addr) {
#ifdef HAVE_G2D
      // Convert source to RGBA for G2D input
      Image rgba_src = ConvertToRGBA(img);

      // Time G2D resize
      auto resize_start = std::chrono::high_resolution_clock::now();

      // Use G2D to resize
      if (!G2DResize(rgba_src.data.data(), rgba_src.width, rgba_src.height,
                     rgba_src.channels, &g2d_output, config.letterbox, 0x808080)) {
        std::cerr << "G2D resize failed, using software fallback\n";
        g2d_available = false;
      } else {
        auto resize_end = std::chrono::high_resolution_clock::now();
        preprocess_resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();
        std::cout << "G2D resize complete: " << std::fixed << std::setprecision(2)
                  << preprocess_resize_ms << " ms\n";

        // For zero-copy, copy G2D output to DMA-BUF (simulates camera pipeline)
        // In a real camera pipeline, this copy wouldn't exist - camera would
        // output directly to a DMA-BUF shared with the inference engine.
        if (use_zerocopy && dmabuf_ptr) {
          auto copy_start = std::chrono::high_resolution_clock::now();
          SyncDmaBufForCpu(dmabuf_fd, true);
          std::memcpy(dmabuf_ptr, g2d_output.virt_addr, g2d_output.size);
          SyncDmaBufForDevice(dmabuf_fd);
          auto copy_end = std::chrono::high_resolution_clock::now();
          preprocess_copy_to_dmabuf_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
          std::cout << "G2D → DMA-BUF copy (one-time setup): " << std::fixed << std::setprecision(2)
                    << preprocess_copy_to_dmabuf_ms << " ms\n";
        }
      }
#endif
    }

    if (!g2d_available) {
      // Software fallback - time it
      auto resize_start = std::chrono::high_resolution_clock::now();
      processed_image = ResizeImage(img, input_width, input_height, config.letterbox, 0x808080);
      if (use_rgba) {
        processed_image = ConvertToRGBA(processed_image);
      }
      auto resize_end = std::chrono::high_resolution_clock::now();
      preprocess_resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();
      std::cout << "Software resize complete: " << std::fixed << std::setprecision(2)
                << preprocess_resize_ms << " ms\n";

      // For zero-copy, copy to DMA-BUF
      if (use_zerocopy && dmabuf_ptr) {
        auto copy_start = std::chrono::high_resolution_clock::now();
        SyncDmaBufForCpu(dmabuf_fd, true);
        std::memcpy(dmabuf_ptr, processed_image.data.data(), processed_image.GetByteSize());
        SyncDmaBufForDevice(dmabuf_fd);
        auto copy_end = std::chrono::high_resolution_clock::now();
        preprocess_copy_to_dmabuf_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
        std::cout << "Copy to DMA-BUF: " << std::fixed << std::setprecision(2)
                  << preprocess_copy_to_dmabuf_ms << " ms\n";
      }
    }
  }

  // Copy to input tensor for non-zerocopy modes
  // Time the copy/alpha-strip operation
  double preprocess_copy_to_tensor_ms = 0.0;

  if (!use_zerocopy) {
    auto copy_start = std::chrono::high_resolution_clock::now();

    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    const uint8_t* src_data = g2d_available
        ? static_cast<const uint8_t*>(g2d_output.virt_addr)
        : processed_image.data.data();
    size_t src_size = g2d_available ? g2d_output.size : processed_image.GetByteSize();

    // Debug: check if tensor data pointer is valid
    if (config.verbose) {
      std::cout << "DEBUG: input_tensor->data.data=" << (void*)input_tensor->data.data
                << " type=" << input_tensor->type
                << " bytes=" << input_tensor->bytes
                << " src_data=" << (void*)src_data
                << " src_size=" << src_size << "\n";
    }

    if (use_rgba) {
      // RGBA -> RGB conversion (drop alpha) with INT8 conversion if needed
      int num_pixels = input_width * input_height;
      if (input_tensor->type == kTfLiteInt8) {
        // INT8 model: UINT8 RGBA → drop alpha + subtract 128 → INT8 RGB
        int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
        ConvertRgbaUint8ToRgbInt8(src_data, dst, num_pixels);
      } else if (input_tensor->type == kTfLiteUInt8) {
        uint8_t* dst = input_tensor->data.uint8;
        ConvertRgbaUint8ToRgbUint8(src_data, dst, num_pixels);
      } else if (input_tensor->type == kTfLiteFloat32) {
        float* dst = interpreter->typed_input_tensor<float>(0);
        for (int i = 0; i < num_pixels; ++i) {
          dst[i * 3 + 0] = src_data[i * 4 + 0] / 255.0f;
          dst[i * 3 + 1] = src_data[i * 4 + 1] / 255.0f;
          dst[i * 3 + 2] = src_data[i * 4 + 2] / 255.0f;
        }
      }
    } else {
      // Direct RGB copy with INT8 conversion if needed
      if (input_tensor->type == kTfLiteInt8) {
        // INT8 model: UINT8 RGB → subtract 128 → INT8 RGB
        int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
        size_t count = std::min(src_size, static_cast<size_t>(input_tensor->bytes));
        ConvertUint8ToInt8(src_data, dst, count);
      } else if (input_tensor->type == kTfLiteUInt8) {
        std::memcpy(input_tensor->data.uint8, src_data,
                    std::min(src_size, static_cast<size_t>(input_tensor->bytes)));
      } else if (input_tensor->type == kTfLiteFloat32) {
        float* dst = interpreter->typed_input_tensor<float>(0);
        size_t count = input_width * input_height * 3;
        for (size_t i = 0; i < count; ++i) {
          dst[i] = src_data[i] / 255.0f;
        }
      }
    }

    auto copy_end = std::chrono::high_resolution_clock::now();
    preprocess_copy_to_tensor_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    std::cout << "Copy to tensor" << (use_rgba ? " (with alpha strip)" : "")
              << ": " << std::fixed << std::setprecision(2)
              << preprocess_copy_to_tensor_ms << " ms\n";
  }

  // Warmup (just invoke, preprocessing already done once above)
  std::cout << "\nRunning " << config.warmup << " warmup iterations...\n";
  for (int i = 0; i < config.warmup; ++i) {
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Warmup failed at iteration " << i << std::endl;
      break;
    }
  }

  // Benchmark with per-iteration timing breakdown
  // Simulates a camera pipeline where each frame goes through:
  // 1. G2D resize (camera frame → model input size)
  // 2. Copy path: cache invalidate + mmap + memcpy to tensor
  //    OR Zerocopy path: G2D finish (no CPU copy)
  // 3. NPU inference
  std::cout << "Running " << config.iterations << " timed iterations (full pipeline)...\n";
  std::vector<double> total_times;
  std::vector<double> resize_times;
  std::vector<double> copy_times;
  std::vector<double> invoke_times;

  // Get pointers for per-iteration operations
  TfLiteTensor* bench_input_tensor = !use_zerocopy ? interpreter->input_tensor(0) : nullptr;

  // Source image for G2D resize (convert once, reuse)
  Image rgba_src;
  if (g2d_available) {
#ifdef HAVE_G2D
    rgba_src = ConvertToRGBA(img);
#endif
  }

  // Create persistent G2D context (reuses handle + source buffer across iterations)
  G2DContext g2d_ctx = {};
  bool have_g2d_ctx = false;
  if (g2d_available) {
#ifdef HAVE_G2D
    if (G2DContextOpen(&g2d_ctx)) {
      if (G2DContextAllocSource(&g2d_ctx, rgba_src.width, rgba_src.height, G2D_RGBA8888)) {
        have_g2d_ctx = true;
        std::cout << "G2D context: handle reused, source buffer "
                  << rgba_src.width << "x" << rgba_src.height << " pre-allocated\n";
        if (use_zerocopy && dmabuf_fd >= 0) {
          if (G2DContextCachePhysAddr(&g2d_ctx, dmabuf_fd)) {
            std::cout << "G2D context: DMA-BUF phys addr cached (0x"
                      << std::hex << g2d_ctx.dmabuf_phys << std::dec << ")\n";
          } else {
            std::cerr << "G2D context: failed to cache DMA-BUF phys addr\n";
          }
        }
      } else {
        std::cerr << "G2D context: failed to allocate source buffer\n";
        G2DContextClose(&g2d_ctx);
      }
    } else {
      std::cerr << "G2D context: failed to open, falling back to per-call\n";
    }
#endif
  }

  // Pre-fill letterbox padding ONCE at init (grey=0x808080)
  // This avoids per-frame fill overhead since letterbox region is constant
  bool letterbox_prefilled = false;
  if (config.letterbox && g2d_available) {
#ifdef HAVE_G2D
    LetterboxInfo lb = CalculateLetterbox(img.width, img.height, input_width, input_height);
    if (lb.offset_x > 0 || lb.offset_y > 0) {
      std::cout << "Pre-filling letterbox padding (grey 0x808080)...\n";
      std::cout << "  Letterbox region: " << lb.scaled_width << "x" << lb.scaled_height
                << " at offset (" << lb.offset_x << ", " << lb.offset_y << ")\n";

      // Pre-fill DMA-BUF for zero-copy path
      if (use_zerocopy && dmabuf_ptr) {
        uint32_t dst_format = use_rgba ? G2D_RGBA8888 : G2D_RGB888;
        G2DFillDmaBuf(dmabuf_ptr, input_width, input_height, dst_format, 0x808080);
        // Flush CPU-written letterbox pixels from cache before G2D/NPU access
        SyncDmaBufForDevice(dmabuf_fd);
        letterbox_prefilled = true;
      }
      // Pre-fill G2D buffer for copy path
      if (!use_zerocopy && g2d_output.virt_addr) {
        G2DFillBuffer(&g2d_output, 0x808080);
        letterbox_prefilled = true;
      }
    }
#endif
  }

  for (int i = 0; i < config.iterations; ++i) {
    auto total_start = std::chrono::high_resolution_clock::now();

    double resize_ms = 0.0;
    double copy_ms = 0.0;
    double invoke_ms = 0.0;

    // STAGE 1 + 2 for zerocopy: G2D resize directly to DMA-BUF (true zero-copy)
    if (use_zerocopy && g2d_available && dmabuf_ptr) {
#ifdef HAVE_G2D
      auto resize_start = std::chrono::high_resolution_clock::now();

      uint32_t dst_format = use_rgba ? G2D_RGBA8888 : G2D_RGB888;
      bool zerocopy_ok;

      if (have_g2d_ctx) {
        // Context path: reuses handle, source buffer, and cached phys addr
        zerocopy_ok = G2DResizeToDmaBufCtx(&g2d_ctx, rgba_src.data.data(),
                                            rgba_src.width, rgba_src.height,
                                            rgba_src.channels, dmabuf_ptr,
                                            input_width, input_height, dst_format,
                                            config.letterbox, 0x808080, letterbox_prefilled);
      } else {
        // Fallback: per-call alloc/free (original behavior)
        zerocopy_ok = G2DResizeToDmaBuf(rgba_src.data.data(), rgba_src.width, rgba_src.height,
                                         rgba_src.channels, dmabuf_fd, dmabuf_ptr,
                                         input_width, input_height, dst_format,
                                         config.letterbox, 0x808080, letterbox_prefilled);
      }

      auto resize_end = std::chrono::high_resolution_clock::now();
      resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();

      // For true zero-copy, copy_ms is 0 (no CPU copy needed, G2D writes directly to DMA-BUF)
      copy_ms = 0.0;

      if (!zerocopy_ok) {
        std::cerr << "G2DResizeToDmaBuf failed at iteration " << i << std::endl;
      }
#endif
    }
    // STAGE 1 for copy path: G2D resize to intermediate buffer
    else if (g2d_available) {
#ifdef HAVE_G2D
      auto resize_start = std::chrono::high_resolution_clock::now();

      if (have_g2d_ctx) {
        // Context path: reuses handle and source buffer
        G2DResizeCtx(&g2d_ctx, rgba_src.data.data(), rgba_src.width, rgba_src.height,
                     rgba_src.channels, &g2d_output, config.letterbox, 0x808080, letterbox_prefilled);
      } else {
        // Fallback: per-call alloc/free (original behavior)
        G2DResize(rgba_src.data.data(), rgba_src.width, rgba_src.height,
                  rgba_src.channels, &g2d_output, config.letterbox, 0x808080, letterbox_prefilled);
      }

      auto resize_end = std::chrono::high_resolution_clock::now();
      resize_ms = std::chrono::duration<double, std::milli>(resize_end - resize_start).count();
#endif
    }

    // STAGE 2 for copy path: Data transfer to model input
    if (!use_zerocopy) {
      auto copy_start = std::chrono::high_resolution_clock::now();

      if (bench_input_tensor && g2d_available && g2d_output.virt_addr) {
        // Copy path: cache invalidate (already done by G2D) + mmap read + copy to tensor
        const uint8_t* src = static_cast<const uint8_t*>(g2d_output.virt_addr);
        int num_pixels = input_width * input_height;

        if (use_rgba) {
          // RGBA -> RGB (drop alpha) with INT8 conversion if needed
          if (bench_input_tensor->type == kTfLiteInt8) {
            // INT8 model: UINT8 RGBA → drop alpha + subtract 128 → INT8 RGB
            int8_t* dst = reinterpret_cast<int8_t*>(bench_input_tensor->data.int8);
            ConvertRgbaUint8ToRgbInt8(src, dst, num_pixels);
          } else {
            // UINT8 model: just drop alpha (NEON optimized)
            uint8_t* dst = bench_input_tensor->data.uint8;
            ConvertRgbaUint8ToRgbUint8(src, dst, num_pixels);
          }
        } else {
          // Direct RGB copy with INT8 conversion if needed
          if (bench_input_tensor->type == kTfLiteInt8) {
            // INT8 model: UINT8 RGB → subtract 128 → INT8 RGB
            int8_t* dst = reinterpret_cast<int8_t*>(bench_input_tensor->data.int8);
            size_t count = std::min(g2d_output.size, static_cast<size_t>(bench_input_tensor->bytes));
            ConvertUint8ToInt8(src, dst, count);
          } else if (bench_input_tensor->type == kTfLiteUInt8) {
            std::memcpy(bench_input_tensor->data.uint8, src,
                        std::min(g2d_output.size, static_cast<size_t>(bench_input_tensor->bytes)));
          }
        }
      }

      auto copy_end = std::chrono::high_resolution_clock::now();
      copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    }

    // STAGE 3: NPU Inference
    auto invoke_start = std::chrono::high_resolution_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Inference failed at iteration " << i << std::endl;
      break;
    }
    auto invoke_end = std::chrono::high_resolution_clock::now();
    invoke_ms = std::chrono::duration<double, std::milli>(invoke_end - invoke_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    total_times.push_back(total_ms);
    resize_times.push_back(resize_ms);
    copy_times.push_back(copy_ms);
    invoke_times.push_back(invoke_ms);

    if (config.verbose) {
      std::cout << "  Iteration " << (i + 1) << ": total=" << std::fixed
                << std::setprecision(2) << total_ms << "ms, resize=" << resize_ms
                << "ms, copy=" << copy_ms << "ms, invoke=" << invoke_ms << "ms\n";
    }
  }

  // Calculate statistics
  if (!total_times.empty()) {
    double sum = std::accumulate(total_times.begin(), total_times.end(), 0.0);
    result.mean_ms = sum / total_times.size();
    result.min_ms = *std::min_element(total_times.begin(), total_times.end());
    result.max_ms = *std::max_element(total_times.begin(), total_times.end());

    double sq_sum = 0.0;
    for (double t : total_times) {
      sq_sum += (t - result.mean_ms) * (t - result.mean_ms);
    }
    result.std_ms = std::sqrt(sq_sum / total_times.size());

    // Per-stage timings
    double resize_sum = std::accumulate(resize_times.begin(), resize_times.end(), 0.0);
    result.resize_mean_ms = resize_sum / resize_times.size();

    double copy_sum = std::accumulate(copy_times.begin(), copy_times.end(), 0.0);
    result.copy_mean_ms = copy_sum / copy_times.size();

    double invoke_sum = std::accumulate(invoke_times.begin(), invoke_times.end(), 0.0);
    result.invoke_mean_ms = invoke_sum / invoke_times.size();

    std::cout << "\n=== Timing Breakdown (per frame) ===" << std::endl;
    std::cout << "  G2D resize:     " << std::fixed << std::setprecision(2)
              << result.resize_mean_ms << " ms"
              << (use_zerocopy ? " (→ DMA-BUF, true zero-copy)" : "") << std::endl;
    std::cout << "  Data transfer:  " << result.copy_mean_ms << " ms"
              << (use_zerocopy ? " (none - G2D wrote directly to DMA-BUF)" :
                  (use_rgba ? " (cache inv + mmap + RGBA->RGB strip)" : " (cache inv + mmap + memcpy)"))
              << std::endl;
    std::cout << "  NPU invoke:     " << result.invoke_mean_ms << " ms"
              << (use_adaptor ? " (includes Slice for RGBA->RGB)" : "") << std::endl;
    std::cout << "  -----------------------------------------" << std::endl;

    // Timing verification: sum of components should match total (within 1%)
    double sum_of_components = result.resize_mean_ms + result.copy_mean_ms + result.invoke_mean_ms;
    double timing_gap = result.mean_ms - sum_of_components;
    double timing_gap_pct = (timing_gap / result.mean_ms) * 100.0;

    std::cout << "  Total mean:     " << result.mean_ms << " ms (min="
              << result.min_ms << ", max=" << result.max_ms << ", std=" << result.std_ms << ")\n";
    std::cout << "  Components sum: " << std::setprecision(2) << sum_of_components
              << " ms (gap=" << std::setprecision(3) << timing_gap << " ms, "
              << std::setprecision(1) << timing_gap_pct << "%)\n";
  }

  // Process results
  if (config.task == "classification") {
    result.results_valid = RunClassificationTest(config, interpreter.get(), labels);
  } else if (config.task == "detection") {
    result.results_valid = RunDetectionTest(config, interpreter.get(), labels,
                                             input_width, input_height);
  }

  // Cleanup G2D context
  if (have_g2d_ctx) {
    G2DContextFreeSource(&g2d_ctx);
    G2DContextClose(&g2d_ctx);
  }

  // Cleanup
  if (buffer_handle != kTfLiteNullBufferHandle) {
    VxDelegateUnregisterDmaBuf(delegate, buffer_handle);
  }
  FreeDmaBuf(dmabuf_fd, dmabuf_ptr, buffer_size);
  G2DFreeBuffer(&g2d_output);
  vx::delegate::VxDelegateDelete(delegate);

  return result;
}

void PrintBenchmarkSummary(const std::vector<BenchmarkResult>& results) {
  std::cout << "\n\n";
  std::cout << "================================================================================\n";
  std::cout << "                           BENCHMARK SUMMARY\n";
  std::cout << "================================================================================\n\n";

  // Main timing table
  std::cout << std::left << std::setw(20) << "Mode"
            << std::right << std::setw(12) << "Total (ms)"
            << std::setw(12) << "Min (ms)"
            << std::setw(12) << "Max (ms)"
            << std::setw(12) << "Std (ms)"
            << std::setw(10) << "Valid"
            << "\n";
  std::cout << std::string(78, '-') << "\n";

  double baseline = 0.0;
  for (const auto& r : results) {
    if (baseline == 0.0) baseline = r.mean_ms;

    std::cout << std::left << std::setw(20) << DataPathModeToString(r.mode)
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << r.mean_ms
              << std::setw(12) << r.min_ms
              << std::setw(12) << r.max_ms
              << std::setw(12) << r.std_ms
              << std::setw(10) << (r.results_valid ? "PASS" : "FAIL")
              << "\n";
  }

  // Timing breakdown table
  std::cout << "\n\n";
  std::cout << "                           TIMING BREAKDOWN\n";
  std::cout << std::string(78, '-') << "\n";
  std::cout << std::left << std::setw(20) << "Mode"
            << std::right << std::setw(14) << "Resize (ms)"
            << std::setw(14) << "Copy (ms)"
            << std::setw(14) << "Invoke (ms)"
            << std::setw(16) << "Alpha Strip"
            << "\n";
  std::cout << std::string(78, '-') << "\n";

  for (const auto& r : results) {
    bool is_rgba_mode = (r.mode == DataPathMode::CopyRgba || r.mode == DataPathMode::ZeroCopyRgba);
    bool is_zerocopy = (r.mode == DataPathMode::ZeroCopyRgb || r.mode == DataPathMode::ZeroCopyRgba);

    std::string alpha_strip;
    if (r.mode == DataPathMode::CopyRgba) {
      alpha_strip = "CPU";
    } else if (r.mode == DataPathMode::ZeroCopyRgba) {
      alpha_strip = "NPU (Slice)";
    } else {
      alpha_strip = "N/A";
    }

    std::cout << std::left << std::setw(20) << DataPathModeToString(r.mode)
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(14) << r.resize_mean_ms
              << std::setw(14) << r.copy_mean_ms
              << std::setw(14) << r.invoke_mean_ms
              << std::setw(16) << alpha_strip
              << "\n";
  }

  std::cout << "\n";

  // Speedup comparison
  if (results.size() > 1 && baseline > 0) {
    std::cout << "Speedup vs " << DataPathModeToString(results[0].mode) << ":\n";
    for (size_t i = 1; i < results.size(); ++i) {
      double speedup = baseline / results[i].mean_ms;
      std::cout << "  " << std::left << std::setw(18) << DataPathModeToString(results[i].mode)
                << ": " << std::fixed << std::setprecision(2) << speedup << "x"
                << " (" << (speedup >= 1.0 ? "+" : "")
                << static_cast<int>((speedup - 1.0) * 100) << "%)\n";
    }
  }

  // Copy overhead analysis
  std::cout << "\nCopy Overhead Analysis:\n";
  double zerocopy_rgb_copy = 0.0;
  double copy_rgb_copy = 0.0;
  double copy_rgba_copy = 0.0;
  double zerocopy_rgba_invoke = 0.0;
  double zerocopy_rgb_invoke = 0.0;

  for (const auto& r : results) {
    switch (r.mode) {
      case DataPathMode::CopyRgb:
        copy_rgb_copy = r.copy_mean_ms;
        break;
      case DataPathMode::CopyRgba:
        copy_rgba_copy = r.copy_mean_ms;
        break;
      case DataPathMode::ZeroCopyRgb:
        zerocopy_rgb_copy = r.copy_mean_ms;
        zerocopy_rgb_invoke = r.invoke_mean_ms;
        break;
      case DataPathMode::ZeroCopyRgba:
        zerocopy_rgba_invoke = r.invoke_mean_ms;
        break;
    }
  }

  if (copy_rgb_copy > 0 && zerocopy_rgb_copy >= 0) {
    std::cout << "  RGB memcpy eliminated: " << std::fixed << std::setprecision(2)
              << (copy_rgb_copy - zerocopy_rgb_copy) << " ms saved\n";
  }
  if (copy_rgba_copy > 0 && zerocopy_rgb_copy >= 0) {
    std::cout << "  RGBA alpha strip (CPU): " << std::fixed << std::setprecision(2)
              << copy_rgba_copy << " ms\n";
  }
  if (zerocopy_rgba_invoke > 0 && zerocopy_rgb_invoke > 0) {
    double npu_slice_overhead = zerocopy_rgba_invoke - zerocopy_rgb_invoke;
    std::cout << "  RGBA alpha strip (NPU Slice overhead): " << std::fixed << std::setprecision(2)
              << npu_slice_overhead << " ms\n";
    if (copy_rgba_copy > 0 && npu_slice_overhead > 0) {
      std::cout << "  NPU Slice vs CPU strip: " << std::fixed << std::setprecision(2)
                << (copy_rgba_copy / npu_slice_overhead) << "x faster on NPU\n";
    }
  }

  std::cout << "\n================================================================================\n";
}

int main(int argc, char* argv[]) {
  Config config = ParseArgs(argc, argv);

  if (config.model_path.empty() || config.image_path.empty()) {
    std::cerr << "Error: --model and --image are required\n\n";
    PrintUsage(argv[0]);
    return 1;
  }

  std::cout << "CameraAdaptor Integration Test\n"
            << "==============================\n"
            << "Model:      " << config.model_path << "\n"
            << "Image:      " << config.image_path << "\n"
            << "Task:       " << config.task << "\n"
            << "Iterations: " << config.iterations << " (+" << config.warmup << " warmup)\n"
            << "Letterbox:  " << (config.letterbox ? "yes" : "no") << "\n"
            << "G2D:        " << (config.use_g2d ? "enabled" : "disabled") << "\n";

  // Check G2D availability
  if (config.use_g2d) {
    if (G2DIsAvailable()) {
      std::cout << "G2D:        available\n";
    } else {
      std::cout << "G2D:        not available (using software resize)\n";
      config.use_g2d = false;
    }
  }

  // Probe and print DMA heap availability
  std::cout << "Heap:       " << HeapTypeToString(config.heap) << "\n";
  ProbeDmaHeaps();

  // Load labels
  std::vector<std::string> labels;
  if (!config.labels_path.empty()) {
    labels = LoadLabels(config.labels_path);
    std::cout << "Labels:     " << labels.size() << " loaded\n";
  }

  std::cout << std::endl;

  if (config.benchmark) {
    // Run all modes
    std::vector<DataPathMode> modes = {
        DataPathMode::CopyRgb,
        DataPathMode::CopyRgba,
        DataPathMode::ZeroCopyRgb,
        DataPathMode::ZeroCopyRgba,
    };

    std::vector<BenchmarkResult> results;
    for (auto mode : modes) {
      BenchmarkResult r = RunModeTest(config, mode, labels);
      results.push_back(r);
    }

    PrintBenchmarkSummary(results);

    // Check if all passed
    bool all_valid = std::all_of(results.begin(), results.end(),
                                  [](const BenchmarkResult& r) { return r.results_valid; });
    return all_valid ? 0 : 1;
  } else {
    // Run single mode
    BenchmarkResult result = RunModeTest(config, config.mode, labels);

    std::cout << "\n" << (result.results_valid ? "[TEST PASSED]" : "[TEST FAILED]") << std::endl;
    return result.results_valid ? 0 : 1;
  }
}
