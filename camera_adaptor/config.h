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

#ifndef EDGEFIRST_CAMERA_ADAPTOR_CONFIG_H_
#define EDGEFIRST_CAMERA_ADAPTOR_CONFIG_H_

#include <cstdint>

#include "color_space.h"

namespace edgefirst {
namespace camera_adaptor {

/**
 * Data type enum for tensor element types.
 *
 * Matches common ML framework data types.
 */
enum class DType {
  Float32,
  Float16,
  Uint8,
  Int8,
};

/**
 * Configuration for camera format adaptation.
 *
 * This is the runtime counterpart to the EdgeFirst Python CameraAdaptor library's
 * configuration. It specifies the camera's native format (adaptor) and the model's
 * expected format (model_format), along with data type and layout information.
 */
struct CameraAdaptorConfig {
  /**
   * Camera format (input to preprocessing).
   * This is the format that the camera/video source provides.
   */
  ColorSpace adaptor = ColorSpace::Rgb;

  /**
   * Model format (output of preprocessing).
   * This is the format that the ML model expects.
   * Typically RGB for most computer vision models.
   */
  ColorSpace model_format = ColorSpace::Rgb;

  /**
   * Input data type (camera tensor type).
   */
  DType input_dtype = DType::Float32;

  /**
   * Output data type (model input tensor type).
   */
  DType output_dtype = DType::Float32;

  /**
   * Whether input uses planar layout (channels in separate planes).
   * If false, input is packed (interleaved channels).
   * Note: This is independent of ColorSpace's inherent layout (e.g., NV12 is always semi-planar).
   */
  bool is_planar = false;

  /**
   * Target resize width (0 = no resize).
   * If non-zero, input will be resized to this width.
   */
  uint32_t resize_width = 0;

  /**
   * Target resize height (0 = no resize).
   * If non-zero, input will be resized to this height.
   */
  uint32_t resize_height = 0;

  /**
   * Apply letterbox padding when resizing.
   * If true, aspect ratio is preserved and padding is added.
   * If false, image is stretched to fit target size.
   */
  bool letterbox = false;

  /**
   * Letterbox padding color as packed RGB (e.g., 0x808080 for grey).
   */
  uint32_t letterbox_color = 0;

  /**
   * Get the number of input channels based on adaptor format.
   */
  int input_channels() const { return GetInputChannels(adaptor); }

  /**
   * Get the number of output channels based on adaptor format.
   */
  int output_channels() const { return GetOutputChannels(adaptor); }

  /**
   * Check if the configuration involves quantized types.
   */
  bool is_quantized() const {
    return input_dtype == DType::Uint8 || input_dtype == DType::Int8 ||
           output_dtype == DType::Uint8 || output_dtype == DType::Int8;
  }

  /**
   * Check if resize is enabled.
   */
  bool has_resize() const {
    return resize_width > 0 && resize_height > 0;
  }
};

/**
 * Create a CameraAdaptorConfig from a FourCC string.
 *
 * @param fourcc FourCC string (e.g., "RGBA", "YUYV", "NV12")
 * @return Config with adaptor set from FourCC, model_format defaults to RGB
 */
CameraAdaptorConfig CreateConfigFromFourCC(const char* fourcc);

/**
 * Create a CameraAdaptorConfig from a V4L2-style FourCC code.
 *
 * @param fourcc V4L2 FourCC code (uint32_t)
 * @return Config with adaptor set from FourCC, model_format defaults to RGB
 */
CameraAdaptorConfig CreateConfigFromFourCC(uint32_t fourcc);

/**
 * Create a CameraAdaptorConfig from a format string.
 *
 * Accepts lowercase format names (e.g., "rgba", "yuyv") or FourCC codes.
 *
 * @param format Format string
 * @return Config with adaptor set from format, model_format defaults to RGB
 */
CameraAdaptorConfig CreateConfigFromString(const char* format);

}  // namespace camera_adaptor
}  // namespace edgefirst

#endif  // EDGEFIRST_CAMERA_ADAPTOR_CONFIG_H_
