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

#ifndef EDGEFIRST_CAMERA_ADAPTOR_CAMERA_ADAPTOR_H_
#define EDGEFIRST_CAMERA_ADAPTOR_CAMERA_ADAPTOR_H_

#include <memory>
#include <string>

#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"

#include "config.h"

namespace edgefirst {
namespace camera_adaptor {

/**
 * Result of preprocessing layer injection.
 *
 * Contains the camera input tensor (new graph input accepting camera format)
 * and the model input tensor (connected to original model operations).
 */
struct PreprocessingResult {
  /**
   * The new input tensor accepting camera format.
   * This replaces the original graph input and should be used for DMA-BUF binding.
   */
  std::shared_ptr<tim::vx::Tensor> camera_input;

  /**
   * The tensor connecting to the original model input.
   * This is the output of the preprocessing operations, connected to the
   * model's first operations.
   */
  std::shared_ptr<tim::vx::Tensor> model_input;

  /**
   * Success status.
   * True if preprocessing was successfully injected.
   */
  bool success = false;

  /**
   * Error message if success is false.
   */
  std::string error_message;
};

/**
 * CameraAdaptor injects preprocessing operations into TIM-VX graphs.
 *
 * This is the runtime counterpart to the EdgeFirst Python CameraAdaptor library.
 * While the Python library enables training models with knowledge of target camera
 * formats, this class injects preprocessing layers at inference time to handle
 * format conversions on the NPU.
 *
 * **Design Principle**: This implementation is TIM-VX specific - no abstraction
 * layers or backend selection. The concept is portable (could be implemented for
 * OpenGLES, TensorRT, CUDA), but this implementation uses TIM-VX/OpenVX types directly.
 *
 * **Usage**:
 * 1. Create a CameraAdaptorConfig specifying camera and model formats
 * 2. Construct a CameraAdaptor with the config
 * 3. Call InjectPreprocessing() with the graph and original input tensor
 * 4. Use the returned camera_input tensor for DMA-BUF binding
 *
 * **Supported Conversions** (this session):
 * - RGBA -> RGB (alpha channel removal via Slice)
 *
 * **Future Conversions** (one per session for focused attention):
 * - BGR <-> RGB swap
 * - YUYV -> RGB conversion
 * - NV12 -> RGB conversion
 * - Bayer demosaic
 * - Resize with letterbox
 */
class CameraAdaptor {
 public:
  /**
   * Construct a CameraAdaptor with the given configuration.
   *
   * @param config Configuration specifying camera and model formats
   */
  explicit CameraAdaptor(const CameraAdaptorConfig& config);

  /**
   * Get the current configuration.
   */
  const CameraAdaptorConfig& config() const { return config_; }

  /**
   * Get the camera adaptor format (input format).
   */
  ColorSpace adaptor() const { return config_.adaptor; }

  /**
   * Get the model format (output format).
   */
  ColorSpace model_format() const { return config_.model_format; }

  /**
   * Check if input layout is planar.
   */
  bool is_planar() const { return config_.is_planar; }

  /**
   * Check if conversion is needed.
   *
   * Returns false if:
   * - adaptor == model_format (same format)
   * - AND no resize is configured
   *
   * In this case, InjectPreprocessing() returns passthrough tensors.
   *
   * @return true if preprocessing operations are needed
   */
  bool RequiresConversion() const;

  /**
   * Inject preprocessing operations into a TIM-VX graph.
   *
   * This method:
   * 1. Creates a new input tensor in camera format
   * 2. Creates preprocessing operations (slice, color conversion, etc.)
   * 3. Connects the preprocessing output to the original model input
   *
   * @param graph The TIM-VX graph to inject operations into
   * @param original_input The existing input tensor (model's expected format)
   * @param dmabuf_fd Optional DMA-BUF file descriptor. If >= 0, the camera_input
   *                  tensor will be created with DMA-BUF backing for zero-copy I/O.
   * @return PreprocessingResult with camera_input (new graph input) and
   *         model_input (connected to original operations)
   *
   * **Important**: After calling this method:
   * - Bind DMA-BUF to camera_input (not original_input)
   * - The original_input tensor should no longer be used directly
   */
  PreprocessingResult InjectPreprocessing(
      std::shared_ptr<tim::vx::Graph> graph,
      std::shared_ptr<tim::vx::Tensor> original_input,
      int dmabuf_fd = -1);

 private:
  CameraAdaptorConfig config_;

  /**
   * Convert 4-channel input (RGBA/BGRA/RGBX/etc.) to 3-channel output.
   *
   * Supports all combinations:
   * - RGBA → RGB: Slice only
   * - RGBA → BGR: Slice + Reverse
   * - BGRA → BGR: Slice only
   * - BGRA → RGB: Slice + Reverse
   *
   * The Slice operation drops the alpha/padding channel.
   * The Reverse operation swaps R↔B channels along axis 0.
   *
   * @param graph TIM-VX graph
   * @param original_input Original 3-channel model input tensor (TRANSIENT)
   * @param dmabuf_fd DMA-BUF fd for zero-copy, or -1
   * @param needs_channel_swap If true, add Reverse operation for R↔B swap
   * @param input_channels Number of input channels (4 for RGBA/BGRA)
   * @return PreprocessingResult with camera_input and model_input tensors
   */
  PreprocessingResult ConvertWithAlphaDrop(
      std::shared_ptr<tim::vx::Graph> graph,
      std::shared_ptr<tim::vx::Tensor> original_input,
      int dmabuf_fd,
      bool needs_channel_swap,
      int input_channels = 4);

  /**
   * Check if conversion between adaptor and model_format requires channel swap.
   *
   * RGB-ordered inputs (RGBA, RGBX, RGB) need swap when output is BGR.
   * BGR-ordered inputs (BGRA, BGRX, BGR) need swap when output is RGB.
   */
  bool NeedsChannelSwap() const;

  // Future converters:
  // PreprocessingResult ConvertYuyvToRgb(...);
  // PreprocessingResult ConvertNv12ToRgb(...);
  // PreprocessingResult ConvertBayerToRgb(...);
  // PreprocessingResult ApplyResize(...);
};

}  // namespace camera_adaptor
}  // namespace edgefirst

#endif  // EDGEFIRST_CAMERA_ADAPTOR_CAMERA_ADAPTOR_H_
