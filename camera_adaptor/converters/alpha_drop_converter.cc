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

#include "../camera_adaptor.h"

#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/reverse.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/simple_operations.h"
#include "tensorflow/lite/minimal_logging.h"

using tflite::TFLITE_LOG_INFO;

namespace edgefirst {
namespace camera_adaptor {

PreprocessingResult CameraAdaptor::ConvertWithAlphaDrop(
    std::shared_ptr<tim::vx::Graph> graph,
    std::shared_ptr<tim::vx::Tensor> original_input,
    int dmabuf_fd,
    bool needs_channel_swap,
    int input_channels) {
  PreprocessingResult result;

  // Get the original tensor spec (model's expected format: 3-channel)
  const tim::vx::TensorSpec& spec = original_input->GetSpec();
  const auto& shape = spec.shape_;

  // TIM-VX shape is reversed from TFLite NHWC:
  // TFLite NHWC [N, H, W, C] becomes TIM-VX [C, W, H, N]
  // So shape[0] = Channels, shape[1] = Width, shape[2] = Height, shape[3] = Batch

  // Validate tensor has at least 3 dimensions
  if (shape.size() < 3) {
    result.error_message = "Converter expects at least 3D tensor (CWH or CWHN), got " +
                           std::to_string(shape.size()) + "D";
    return result;
  }

  // Validate model expects 3 channels (RGB or BGR)
  uint32_t expected_output_channels = shape[0];
  if (expected_output_channels != 3) {
    result.error_message = "Converter expects model input to have 3 channels, got " +
                           std::to_string(expected_output_channels);
    return result;
  }

  // Check if model expects INT8 but camera provides UINT8
  // This requires a subtract-128 operation to convert UINT8 [0,255] to INT8 [-128,127]
  bool needs_uint8_to_int8 = (spec.datatype_ == tim::vx::DataType::INT8);

  // Create camera input tensor spec (4 channels: RGBA, BGRA, etc.)
  // For INT8 models, camera input is UINT8 with zero_point=0 quantization
  auto camera_shape = shape;
  camera_shape[0] = static_cast<uint32_t>(input_channels);

  tim::vx::DataType camera_datatype = spec.datatype_;
  tim::vx::Quantization camera_quant = spec.quantization_;

  if (needs_uint8_to_int8) {
    // Camera provides UINT8 data [0-255]
    // Use UINT8 datatype with zero_point=0 for camera input
    camera_datatype = tim::vx::DataType::UINT8;
    // Quantization: same scale as model, but zero_point=0 for UINT8
    // The model's quantization typically has zero_point=-128 for INT8
    // Camera UINT8 with zp=0: float = (uint8 - 0) * scale = uint8 * scale
    camera_quant = tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC,
                                          spec.quantization_.Scales()[0],
                                          0);  // zero_point=0 for UINT8
  }

  tim::vx::TensorSpec camera_spec(
      camera_datatype,
      camera_shape,
      tim::vx::TensorAttribute::INPUT,
      camera_quant);

  // Create camera input tensor with optional DMA-BUF backing
  std::shared_ptr<tim::vx::Tensor> camera_input;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
  if (dmabuf_fd >= 0) {
    tim::vx::DmaBufferDesc dma_desc;
    dma_desc.fd = static_cast<int64_t>(dmabuf_fd);
    camera_input = graph->CreateTensor(camera_spec, dma_desc);
  } else {
    camera_input = graph->CreateTensor(camera_spec);
  }
#else
  (void)dmabuf_fd;
  camera_input = graph->CreateTensor(camera_spec);
#endif

  if (!camera_input) {
    result.error_message = "Failed to create camera input tensor";
    return result;
  }

  // Log the pipeline we're building
  TFLITE_LOG(TFLITE_LOG_INFO, 
             "CameraAdaptor pipeline: input_channels=%d, needs_swap=%d, needs_uint8_to_int8=%d, dmabuf_fd=%d",
             input_channels, needs_channel_swap ? 1 : 0, needs_uint8_to_int8 ? 1 : 0, dmabuf_fd);

  // Build the preprocessing pipeline
  // Pipeline (4ch): camera_input (4ch UINT8) -> Slice (3ch) -> [Reverse if swap] -> [Sub if INT8] -> model_input
  // Pipeline (3ch): camera_input (3ch UINT8) -> [Reverse if swap] -> [Sub if INT8] -> model_input
  //
  // For INT8 models with UINT8 camera input, we need to subtract 128:
  //   UINT8 [0,255] - 128 = INT8 [-128,127]
  //
  // The pipeline uses TRANSIENT tensors between operations.

  // Determine how many intermediate stages we need
  bool has_slice = (input_channels > 3);  // Only needed for 4-channel input
  // needs_channel_swap and needs_uint8_to_int8 are already set

  // Create intermediate tensors based on what operations we need
  // Each intermediate uses UINT8 until the final conversion to INT8
  tim::vx::DataType intermediate_dtype = needs_uint8_to_int8 ? 
      tim::vx::DataType::UINT8 : spec.datatype_;
  tim::vx::Quantization intermediate_quant = needs_uint8_to_int8 ?
      camera_quant : spec.quantization_;

  // Track the current output tensor as we build the pipeline
  std::shared_ptr<tim::vx::Tensor> current_output;

  // Handle Slice if needed (4-channel to 3-channel)
  if (has_slice) {
    // Slice output tensor
    std::shared_ptr<tim::vx::Tensor> slice_output;
    tim::vx::TensorSpec slice_output_spec(
        intermediate_dtype,
        shape,
        tim::vx::TensorAttribute::TRANSIENT,
        intermediate_quant);

    // If we only have Slice (no Reverse, no Sub), output directly to original_input
    if (!needs_channel_swap && !needs_uint8_to_int8) {
      slice_output = original_input;
    } else {
      slice_output = graph->CreateTensor(slice_output_spec);
      if (!slice_output) {
        result.error_message = "Failed to create slice output tensor";
        return result;
      }
    }

    // Create Slice operation: extract channels 0-2 (drop alpha at channel 3)
    std::vector<int32_t> start(shape.size(), 0);
    std::vector<int32_t> length;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == 0) {
        length.push_back(3);  // Extract 3 channels from the 4-channel input
      } else {
        length.push_back(static_cast<int32_t>(camera_shape[i]));
      }
    }

    auto slice = graph->CreateOperation<tim::vx::ops::Slice>(
        static_cast<uint32_t>(shape.size()),
        start,
        length);

    if (!slice) {
      result.error_message = "Failed to create Slice operation";
      return result;
    }

    (*slice).BindInput(camera_input).BindOutput(slice_output);
    TFLITE_LOG(TFLITE_LOG_INFO, "CameraAdaptor: Added Slice op (4ch -> 3ch)");
    current_output = slice_output;
  } else {
    // No Slice needed, start from camera_input directly
    TFLITE_LOG(TFLITE_LOG_INFO, "CameraAdaptor: No Slice needed (3ch input)");
    current_output = camera_input;
  }

  // Add Reverse operation if channel swap needed
  if (needs_channel_swap) {
    std::shared_ptr<tim::vx::Tensor> reverse_output;
    
    // If we still need Sub after this, create intermediate; otherwise output to model
    if (needs_uint8_to_int8) {
      tim::vx::TensorSpec reverse_output_spec(
          intermediate_dtype,
          shape,
          tim::vx::TensorAttribute::TRANSIENT,
          intermediate_quant);
      reverse_output = graph->CreateTensor(reverse_output_spec);
      if (!reverse_output) {
        result.error_message = "Failed to create reverse output tensor";
        return result;
      }
    } else {
      reverse_output = original_input;
    }

    std::vector<int32_t> reverse_axes = {0};
    auto reverse = graph->CreateOperation<tim::vx::ops::Reverse>(reverse_axes);
    if (!reverse) {
      result.error_message = "Failed to create Reverse operation";
      return result;
    }

    (*reverse).BindInput(current_output).BindOutput(reverse_output);
    current_output = reverse_output;
  }

  // Add DataConvert for UINT8->INT8 conversion if needed
  // This converts camera UINT8 data [0,255] to model INT8 input [-128,127].
  //
  // The conversion uses quantization parameters:
  //   Camera input: UINT8, scale=s, zp=0    -> real_value = (uint8 - 0) * s
  //   Model input:  INT8,  scale=s, zp=-128 -> real_value = (int8 + 128) * s
  //
  // For equal real values: uint8 * s = (int8 + 128) * s
  // Therefore: int8 = uint8 - 128
  //
  // DataConvert handles this automatically via the quantization difference:
  //   UINT8(0)   -> INT8(-128)
  //   UINT8(128) -> INT8(0)
  //   UINT8(255) -> INT8(127)
  if (needs_uint8_to_int8) {
    TFLITE_LOG(TFLITE_LOG_INFO, "CameraAdaptor: Adding DataConvert for UINT8->INT8 (quantization handles -128)");
    
    // DataConvert operation: UINT8 (zp=0) -> INT8 (zp=-128)
    // The quantization difference automatically applies the -128 shift
    auto data_convert = graph->CreateOperation<tim::vx::ops::DataConvert>();
    if (!data_convert) {
      result.error_message = "Failed to create DataConvert operation for INT8 conversion";
      return result;
    }

    (*data_convert).BindInput(current_output).BindOutput(original_input);
    TFLITE_LOG(TFLITE_LOG_INFO, 
               "CameraAdaptor: DataConvert bound - UINT8(zp=0) -> INT8(zp=-128)");
    current_output = original_input;
  }

  result.camera_input = camera_input;
  result.model_input = current_output;
  TFLITE_LOG(TFLITE_LOG_INFO, 
             "CameraAdaptor: Pipeline complete - camera_input=%p, model_input=%p",
             camera_input.get(), current_output.get());
  result.success = true;
  return result;
}

}  // namespace camera_adaptor
}  // namespace edgefirst
