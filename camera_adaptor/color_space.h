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

#ifndef EDGEFIRST_CAMERA_ADAPTOR_COLOR_SPACE_H_
#define EDGEFIRST_CAMERA_ADAPTOR_COLOR_SPACE_H_

#include <cstdint>

namespace edgefirst {
namespace camera_adaptor {

/**
 * ColorSpace enum defining supported camera input formats.
 *
 * This is the runtime counterpart to the EdgeFirst Python CameraAdaptor library.
 * Covers common formats from V4L2, GStreamer, and libcamera.
 *
 * Naming conventions:
 * - Lowercase enum names match Python library
 * - Comments include V4L2 FourCC codes
 * - GStreamer format names in parentheses where different
 */
enum class ColorSpace {
  // =========================================================================
  // RGB/BGR Family - Packed formats
  // =========================================================================
  Rgb,      // RGB 24-bit packed (3 bytes/pixel) - V4L2: RGB3 (GStreamer: RGB)
  Bgr,      // BGR 24-bit packed (3 bytes/pixel) - V4L2: BGR3 (GStreamer: BGR)
  Rgba,     // RGBA 32-bit packed (4 bytes/pixel) - V4L2: RGBA (GStreamer: RGBA)
  Bgra,     // BGRA 32-bit packed (4 bytes/pixel) - V4L2: BGRA (GStreamer: BGRA)
  Argb,     // ARGB 32-bit packed (alpha first) - V4L2: ARGB (GStreamer: ARGB)
  Abgr,     // ABGR 32-bit packed (alpha first) - V4L2: ABGR (GStreamer: ABGR)
  Rgbx,     // RGBx 32-bit packed (x=padding, ignored) - V4L2: RGBX (GStreamer: RGBx)
  Bgrx,     // BGRx 32-bit packed (x=padding, ignored) - V4L2: BGRX (GStreamer: BGRx)
  Xrgb,     // xRGB 32-bit packed (x=padding first) - V4L2: XRGB (GStreamer: xRGB)
  Xbgr,     // xBGR 32-bit packed (x=padding first) - V4L2: XBGR (GStreamer: xBGR)
  Rgb565,   // RGB 16-bit packed (5-6-5) - V4L2: RGBP (GStreamer: RGB16)
  Bgr565,   // BGR 16-bit packed (5-6-5) - V4L2: RGBR (GStreamer: BGR16)

  // =========================================================================
  // Grayscale
  // =========================================================================
  Grey,     // 8-bit grayscale (1 byte/pixel) - V4L2: GREY (GStreamer: GRAY8)

  // =========================================================================
  // YUV 4:2:2 - Packed formats (2 bytes/pixel average)
  // =========================================================================
  Yuyv,     // YUV 4:2:2 packed (Y0-U0-Y1-V0) - V4L2: YUYV (GStreamer: YUY2)
  Uyvy,     // YUV 4:2:2 packed (U0-Y0-V0-Y1) - V4L2: UYVY (GStreamer: UYVY)
  Yvyu,     // YUV 4:2:2 packed (Y0-V0-Y1-U0) - V4L2: YVYU (GStreamer: YVYU)
  Vyuy,     // YUV 4:2:2 packed (V0-Y0-U0-Y1) - V4L2: VYUY

  // =========================================================================
  // YUV 4:2:2 - Planar formats
  // =========================================================================
  Yv16,     // YUV 4:2:2 planar (Y, V, U planes) - V4L2: YV16 (GStreamer: YV16)
  I422,     // YUV 4:2:2 planar (Y, U, V planes) - V4L2: 422P (GStreamer: I422/Y42B)

  // =========================================================================
  // YUV 4:2:0 - Semi-planar formats (Y plane + interleaved UV)
  // =========================================================================
  Nv12,     // YUV 4:2:0 semi-planar (Y + UV) - V4L2: NV12 (GStreamer: NV12)
  Nv21,     // YUV 4:2:0 semi-planar (Y + VU) - V4L2: NV21 (GStreamer: NV21)

  // =========================================================================
  // YUV 4:2:0 - Planar formats
  // =========================================================================
  Yv12,     // YUV 4:2:0 planar (Y, V, U planes) - V4L2: YV12 (GStreamer: YV12)
  I420,     // YUV 4:2:0 planar (Y, U, V planes) - V4L2: YU12 (GStreamer: I420)

  // =========================================================================
  // YUV 4:4:4 - Full chroma resolution
  // =========================================================================
  Yuv444,   // YUV 4:4:4 planar - V4L2: YUV4 (GStreamer: Y444)

  // =========================================================================
  // Bayer RAW - Single channel with color filter array
  // =========================================================================
  Rggb,     // Bayer RGGB pattern - V4L2: RGGB (GStreamer: bggr)
  Bggr,     // Bayer BGGR pattern - V4L2: BGGR (GStreamer: rggb)
  Grbg,     // Bayer GRBG pattern - V4L2: GRBG (GStreamer: grbg)
  Gbrg,     // Bayer GBRG pattern - V4L2: GBRG (GStreamer: gbrg)
};

/**
 * Get the number of input channels for a color space.
 *
 * This is the channel count as it appears in the camera input tensor.
 * For example, RGBA has 4 input channels, while NV12 is treated as a
 * special planar format where input channels depends on layout.
 */
int GetInputChannels(ColorSpace cs);

/**
 * Get the number of output channels for a color space.
 *
 * This is the channel count after conversion to the model's expected format.
 * For example, RGBA outputs 3 channels (RGB) after dropping alpha.
 */
int GetOutputChannels(ColorSpace cs);

/**
 * Check if a color space is planar (separate planes per channel).
 *
 * Planar formats have separate memory regions for each channel/component.
 * NV12/NV21 are semi-planar (Y plane + interleaved UV plane).
 */
bool IsPlanar(ColorSpace cs);

/**
 * Check if a color space is packed (interleaved channels).
 *
 * Packed formats have channel data interleaved per-pixel.
 */
bool IsPacked(ColorSpace cs);

/**
 * Get the FourCC code string for a color space.
 *
 * FourCC codes are used for V4L2/GStreamer/libcamera interoperability.
 * @param cs ColorSpace to get FourCC for
 * @return FourCC string (e.g., "RGBA", "NV12"), or nullptr if unsupported
 */
const char* GetFourCC(ColorSpace cs);

/**
 * Get the planar FourCC code string for a color space.
 *
 * Some formats have different FourCC codes for planar layout.
 * @param cs ColorSpace to get planar FourCC for
 * @return Planar FourCC string (e.g., "RGBP", "BGRP"), or nullptr if unsupported
 */
const char* GetPlanarFourCC(ColorSpace cs);

/**
 * Parse a FourCC string to ColorSpace.
 *
 * Accepts both packed and planar FourCC codes.
 * @param fourcc FourCC string (e.g., "RGBA", "YUYV")
 * @return ColorSpace, or Rgb as default if unrecognized
 */
ColorSpace FromFourCC(const char* fourcc);

/**
 * Parse a V4L2-style FourCC code (uint32_t) to ColorSpace.
 *
 * V4L2 encodes FourCC as little-endian uint32_t.
 * @param fourcc V4L2 FourCC code
 * @return ColorSpace, or Rgb as default if unrecognized
 */
ColorSpace FromFourCC(uint32_t fourcc);

/**
 * Convert ColorSpace to string representation.
 *
 * @param cs ColorSpace to convert
 * @return String name (e.g., "rgb", "rgba", "nv12")
 */
const char* ColorSpaceToString(ColorSpace cs);

/**
 * Parse a string to ColorSpace.
 *
 * Accepts lowercase names (e.g., "rgb", "rgba", "nv12").
 * @param str String name
 * @return ColorSpace, or Rgb as default if unrecognized
 */
ColorSpace ColorSpaceFromString(const char* str);

/**
 * Helper to create a V4L2-style FourCC uint32_t from a 4-character string.
 *
 * @param a First character
 * @param b Second character
 * @param c Third character
 * @param d Fourth character
 * @return V4L2 FourCC code
 */
constexpr uint32_t MakeFourCC(char a, char b, char c, char d) {
  return static_cast<uint32_t>(a) |
         (static_cast<uint32_t>(b) << 8) |
         (static_cast<uint32_t>(c) << 16) |
         (static_cast<uint32_t>(d) << 24);
}

}  // namespace camera_adaptor
}  // namespace edgefirst

#endif  // EDGEFIRST_CAMERA_ADAPTOR_COLOR_SPACE_H_
