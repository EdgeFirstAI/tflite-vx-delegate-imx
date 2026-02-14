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

#include "camera_adaptor.h"

namespace edgefirst {
namespace camera_adaptor {

namespace {

// Check if a color space has RGB channel ordering (R at lowest address)
bool IsRgbOrdered(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::Rgb:
    case ColorSpace::Rgba:
    case ColorSpace::Rgbx:
    case ColorSpace::Argb:
    case ColorSpace::Xrgb:
      return true;
    default:
      return false;
  }
}

// Check if a color space has BGR channel ordering (B at lowest address)
bool IsBgrOrdered(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::Bgr:
    case ColorSpace::Bgra:
    case ColorSpace::Bgrx:
    case ColorSpace::Abgr:
    case ColorSpace::Xbgr:
      return true;
    default:
      return false;
  }
}

// Check if a color space has 4 channels (with alpha or padding)
bool Has4Channels(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::Rgba:
    case ColorSpace::Bgra:
    case ColorSpace::Argb:
    case ColorSpace::Abgr:
    case ColorSpace::Rgbx:
    case ColorSpace::Bgrx:
    case ColorSpace::Xrgb:
    case ColorSpace::Xbgr:
      return true;
    default:
      return false;
  }
}

}  // namespace

CameraAdaptor::CameraAdaptor(const CameraAdaptorConfig& config)
    : config_(config) {}

bool CameraAdaptor::RequiresConversion() const {
  // Format conversion always needed if formats differ
  if (config_.adaptor != config_.model_format) {
    return true;
  }
  // Resize always requires conversion
  if (config_.has_resize()) {
    return true;
  }
  // Even if formats match, we may need UINT8→INT8 conversion for INT8 models
  // This is determined later when we have access to the tensor spec
  // For now, return true to allow the dispatch logic to check
  return true;
}

bool CameraAdaptor::NeedsChannelSwap() const {
  bool input_is_rgb = IsRgbOrdered(config_.adaptor);
  bool input_is_bgr = IsBgrOrdered(config_.adaptor);
  bool output_is_rgb = IsRgbOrdered(config_.model_format);
  bool output_is_bgr = IsBgrOrdered(config_.model_format);

  // Swap needed when input and output have different channel ordering
  return (input_is_rgb && output_is_bgr) || (input_is_bgr && output_is_rgb);
}

PreprocessingResult CameraAdaptor::InjectPreprocessing(
    std::shared_ptr<tim::vx::Graph> graph,
    std::shared_ptr<tim::vx::Tensor> original_input,
    int dmabuf_fd) {
  PreprocessingResult result;

  // Validate inputs
  if (!graph) {
    result.error_message = "Graph is null";
    return result;
  }
  if (!original_input) {
    result.error_message = "Original input tensor is null";
    return result;
  }

  // Check if conversion is actually needed
  if (!RequiresConversion()) {
    // Passthrough: return the original tensor as both camera and model input
    result.camera_input = original_input;
    result.model_input = original_input;
    result.success = true;
    return result;
  }

  // Dispatch based on input format
  // 4-channel formats (RGBA, BGRA, RGBX, BGRX, etc.) use alpha-drop converter
  if (Has4Channels(config_.adaptor)) {
    return ConvertWithAlphaDrop(graph, original_input, dmabuf_fd,
                                 NeedsChannelSwap(), 4);
  }

  // 3-channel RGB/BGR formats
  if (config_.adaptor == ColorSpace::Rgb || config_.adaptor == ColorSpace::Bgr) {
    // Check if model expects INT8 but camera provides UINT8
    auto spec = original_input->GetSpec();
    bool needs_uint8_to_int8 = (spec.datatype_ == tim::vx::DataType::INT8);

    if (NeedsChannelSwap()) {
      // TODO: Implement 3-channel RGB↔BGR swap (Reverse only, no Slice)
      result.error_message = "RGB/BGR 3-channel swap not yet implemented";
      return result;
    }

    if (needs_uint8_to_int8) {
      // RGB→RGB but need UINT8→INT8 conversion
      // Use the alpha_drop converter with 3 input channels (no Slice, just Sub)
      return ConvertWithAlphaDrop(graph, original_input, dmabuf_fd,
                                   false /* no channel swap */, 3);
    }

    // Same ordering and data type, true passthrough
    result.camera_input = original_input;
    result.model_input = original_input;
    result.success = true;
    return result;
  }

  // Grayscale
  if (config_.adaptor == ColorSpace::Grey) {
    result.error_message = "Grey to RGB/BGR conversion not yet implemented";
    return result;
  }

  // YUV formats
  switch (config_.adaptor) {
    case ColorSpace::Yuyv:
      result.error_message = "YUYV to RGB/BGR conversion not yet implemented";
      return result;
    case ColorSpace::Uyvy:
      result.error_message = "UYVY to RGB/BGR conversion not yet implemented";
      return result;
    case ColorSpace::Nv12:
      result.error_message = "NV12 to RGB/BGR conversion not yet implemented";
      return result;
    case ColorSpace::Nv21:
      result.error_message = "NV21 to RGB/BGR conversion not yet implemented";
      return result;
    default:
      break;
  }

  // Bayer formats
  switch (config_.adaptor) {
    case ColorSpace::Rggb:
    case ColorSpace::Bggr:
    case ColorSpace::Grbg:
    case ColorSpace::Gbrg:
      result.error_message = "Bayer demosaic not yet implemented";
      return result;
    default:
      break;
  }

  result.error_message = "Unsupported color space conversion";
  return result;
}

}  // namespace camera_adaptor
}  // namespace edgefirst
