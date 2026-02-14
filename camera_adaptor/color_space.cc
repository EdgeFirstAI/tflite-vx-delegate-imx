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

#include "color_space.h"

#include <cstring>
#include <cctype>

namespace edgefirst {
namespace camera_adaptor {

namespace {

// Helper to compare strings case-insensitively
bool StrCaseEqual(const char* a, const char* b) {
  if (!a || !b) return false;
  while (*a && *b) {
    if (std::tolower(static_cast<unsigned char>(*a)) !=
        std::tolower(static_cast<unsigned char>(*b))) {
      return false;
    }
    ++a;
    ++b;
  }
  return *a == *b;
}

}  // namespace

int GetInputChannels(ColorSpace cs) {
  switch (cs) {
    // RGB 24-bit packed (3 bytes/pixel)
    case ColorSpace::Rgb:
    case ColorSpace::Bgr:
      return 3;

    // RGB 32-bit packed (4 bytes/pixel)
    case ColorSpace::Rgba:
    case ColorSpace::Bgra:
    case ColorSpace::Argb:
    case ColorSpace::Abgr:
    case ColorSpace::Rgbx:
    case ColorSpace::Bgrx:
    case ColorSpace::Xrgb:
    case ColorSpace::Xbgr:
      return 4;

    // RGB 16-bit packed (2 bytes/pixel)
    case ColorSpace::Rgb565:
    case ColorSpace::Bgr565:
      return 2;  // Treated as 2-byte packed, not 3 channels

    // Grayscale (1 byte/pixel)
    case ColorSpace::Grey:
      return 1;

    // YUV 4:2:2 packed (2 bytes/pixel average)
    case ColorSpace::Yuyv:
    case ColorSpace::Uyvy:
    case ColorSpace::Yvyu:
    case ColorSpace::Vyuy:
      return 2;

    // YUV 4:2:2 planar (3 planes)
    case ColorSpace::Yv16:
    case ColorSpace::I422:
      return 3;

    // YUV 4:2:0 semi-planar (Y plane + UV plane)
    case ColorSpace::Nv12:
    case ColorSpace::Nv21:
      return 2;  // Y plane + UV plane (treated as 2 components)

    // YUV 4:2:0 planar (3 planes)
    case ColorSpace::Yv12:
    case ColorSpace::I420:
      return 3;

    // YUV 4:4:4 planar (3 planes, full resolution)
    case ColorSpace::Yuv444:
      return 3;

    // Bayer (1 channel raw sensor data)
    case ColorSpace::Rggb:
    case ColorSpace::Bggr:
    case ColorSpace::Grbg:
    case ColorSpace::Gbrg:
      return 1;

    default:
      return 3;
  }
}

int GetOutputChannels(ColorSpace cs) {
  switch (cs) {
    // Grayscale stays as 1 channel
    case ColorSpace::Grey:
      return 1;

    // All other formats convert to RGB (3 channels)
    default:
      return 3;
  }
}

bool IsPlanar(ColorSpace cs) {
  switch (cs) {
    // Semi-planar: Y plane + interleaved UV plane
    case ColorSpace::Nv12:
    case ColorSpace::Nv21:
      return true;

    // Fully planar: Y, U, V separate planes
    case ColorSpace::Yv16:
    case ColorSpace::I422:
    case ColorSpace::Yv12:
    case ColorSpace::I420:
    case ColorSpace::Yuv444:
      return true;

    // All others are packed
    default:
      return false;
  }
}

bool IsPacked(ColorSpace cs) {
  return !IsPlanar(cs);
}

const char* GetFourCC(ColorSpace cs) {
  switch (cs) {
    // RGB/BGR packed
    case ColorSpace::Rgb:
      return "RGB3";
    case ColorSpace::Bgr:
      return "BGR3";
    case ColorSpace::Rgba:
      return "RGBA";
    case ColorSpace::Bgra:
      return "BGRA";
    case ColorSpace::Argb:
      return "ARGB";
    case ColorSpace::Abgr:
      return "ABGR";
    case ColorSpace::Rgbx:
      return "RGBX";
    case ColorSpace::Bgrx:
      return "BGRX";
    case ColorSpace::Xrgb:
      return "XRGB";
    case ColorSpace::Xbgr:
      return "XBGR";
    case ColorSpace::Rgb565:
      return "RGBP";  // V4L2 RGB565
    case ColorSpace::Bgr565:
      return "RGBR";  // V4L2 BGR565

    // Grayscale
    case ColorSpace::Grey:
      return "GREY";

    // YUV 4:2:2 packed
    case ColorSpace::Yuyv:
      return "YUYV";
    case ColorSpace::Uyvy:
      return "UYVY";
    case ColorSpace::Yvyu:
      return "YVYU";
    case ColorSpace::Vyuy:
      return "VYUY";

    // YUV 4:2:2 planar
    case ColorSpace::Yv16:
      return "YV16";
    case ColorSpace::I422:
      return "422P";

    // YUV 4:2:0 semi-planar
    case ColorSpace::Nv12:
      return "NV12";
    case ColorSpace::Nv21:
      return "NV21";

    // YUV 4:2:0 planar
    case ColorSpace::Yv12:
      return "YV12";
    case ColorSpace::I420:
      return "YU12";

    // YUV 4:4:4
    case ColorSpace::Yuv444:
      return "YUV4";

    // Bayer
    case ColorSpace::Rggb:
      return "RGGB";
    case ColorSpace::Bggr:
      return "BGGR";
    case ColorSpace::Grbg:
      return "GRBG";
    case ColorSpace::Gbrg:
      return "GBRG";

    default:
      return nullptr;
  }
}

const char* GetPlanarFourCC(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::Rgb:
    case ColorSpace::Rgba:
    case ColorSpace::Rgbx:
      return "RGBP";
    case ColorSpace::Bgr:
    case ColorSpace::Bgra:
    case ColorSpace::Bgrx:
      return "BGRP";
    default:
      return nullptr;
  }
}

ColorSpace FromFourCC(const char* fourcc) {
  if (!fourcc) return ColorSpace::Rgb;

  // RGB/BGR packed formats
  if (StrCaseEqual(fourcc, "RGB3") || StrCaseEqual(fourcc, "RGB")) {
    return ColorSpace::Rgb;
  }
  if (StrCaseEqual(fourcc, "BGR3") || StrCaseEqual(fourcc, "BGR")) {
    return ColorSpace::Bgr;
  }
  if (StrCaseEqual(fourcc, "RGBA")) return ColorSpace::Rgba;
  if (StrCaseEqual(fourcc, "BGRA")) return ColorSpace::Bgra;
  if (StrCaseEqual(fourcc, "ARGB")) return ColorSpace::Argb;
  if (StrCaseEqual(fourcc, "ABGR")) return ColorSpace::Abgr;
  if (StrCaseEqual(fourcc, "RGBX") || StrCaseEqual(fourcc, "RGBx")) {
    return ColorSpace::Rgbx;
  }
  if (StrCaseEqual(fourcc, "BGRX") || StrCaseEqual(fourcc, "BGRx")) {
    return ColorSpace::Bgrx;
  }
  if (StrCaseEqual(fourcc, "XRGB") || StrCaseEqual(fourcc, "xRGB")) {
    return ColorSpace::Xrgb;
  }
  if (StrCaseEqual(fourcc, "XBGR") || StrCaseEqual(fourcc, "xBGR")) {
    return ColorSpace::Xbgr;
  }
  if (StrCaseEqual(fourcc, "RGBP") || StrCaseEqual(fourcc, "RGB16")) {
    return ColorSpace::Rgb565;
  }
  if (StrCaseEqual(fourcc, "RGBR") || StrCaseEqual(fourcc, "BGR16")) {
    return ColorSpace::Bgr565;
  }

  // Grayscale
  if (StrCaseEqual(fourcc, "GREY") || StrCaseEqual(fourcc, "GRAY") ||
      StrCaseEqual(fourcc, "GRAY8") || StrCaseEqual(fourcc, "Y8")) {
    return ColorSpace::Grey;
  }

  // YUV 4:2:2 packed
  if (StrCaseEqual(fourcc, "YUYV") || StrCaseEqual(fourcc, "YUY2")) {
    return ColorSpace::Yuyv;
  }
  if (StrCaseEqual(fourcc, "UYVY")) return ColorSpace::Uyvy;
  if (StrCaseEqual(fourcc, "YVYU")) return ColorSpace::Yvyu;
  if (StrCaseEqual(fourcc, "VYUY")) return ColorSpace::Vyuy;

  // YUV 4:2:2 planar
  if (StrCaseEqual(fourcc, "YV16")) return ColorSpace::Yv16;
  if (StrCaseEqual(fourcc, "422P") || StrCaseEqual(fourcc, "I422") ||
      StrCaseEqual(fourcc, "Y42B")) {
    return ColorSpace::I422;
  }

  // YUV 4:2:0 semi-planar
  if (StrCaseEqual(fourcc, "NV12")) return ColorSpace::Nv12;
  if (StrCaseEqual(fourcc, "NV21")) return ColorSpace::Nv21;

  // YUV 4:2:0 planar
  if (StrCaseEqual(fourcc, "YV12")) return ColorSpace::Yv12;
  if (StrCaseEqual(fourcc, "YU12") || StrCaseEqual(fourcc, "I420") ||
      StrCaseEqual(fourcc, "IYUV")) {
    return ColorSpace::I420;
  }

  // YUV 4:4:4
  if (StrCaseEqual(fourcc, "YUV4") || StrCaseEqual(fourcc, "Y444")) {
    return ColorSpace::Yuv444;
  }

  // Bayer
  if (StrCaseEqual(fourcc, "RGGB")) return ColorSpace::Rggb;
  if (StrCaseEqual(fourcc, "BGGR")) return ColorSpace::Bggr;
  if (StrCaseEqual(fourcc, "GRBG")) return ColorSpace::Grbg;
  if (StrCaseEqual(fourcc, "GBRG")) return ColorSpace::Gbrg;

  return ColorSpace::Rgb;
}

ColorSpace FromFourCC(uint32_t fourcc) {
  // V4L2 FourCC codes (little-endian)
  switch (fourcc) {
    // RGB/BGR packed
    case MakeFourCC('R', 'G', 'B', '3'):
      return ColorSpace::Rgb;
    case MakeFourCC('B', 'G', 'R', '3'):
      return ColorSpace::Bgr;
    case MakeFourCC('R', 'G', 'B', 'A'):
      return ColorSpace::Rgba;
    case MakeFourCC('B', 'G', 'R', 'A'):
      return ColorSpace::Bgra;
    case MakeFourCC('A', 'R', 'G', 'B'):
      return ColorSpace::Argb;
    case MakeFourCC('A', 'B', 'G', 'R'):
      return ColorSpace::Abgr;
    case MakeFourCC('R', 'G', 'B', 'X'):
      return ColorSpace::Rgbx;
    case MakeFourCC('B', 'G', 'R', 'X'):
      return ColorSpace::Bgrx;
    case MakeFourCC('X', 'R', 'G', 'B'):
      return ColorSpace::Xrgb;
    case MakeFourCC('X', 'B', 'G', 'R'):
      return ColorSpace::Xbgr;
    case MakeFourCC('R', 'G', 'B', 'P'):  // V4L2 RGB565
      return ColorSpace::Rgb565;
    case MakeFourCC('R', 'G', 'B', 'R'):  // V4L2 BGR565
      return ColorSpace::Bgr565;

    // V4L2 32-bit format aliases (different byte order conventions)
    case MakeFourCC('A', 'B', '2', '4'):  // V4L2 ABGR32 (RGBA in memory)
      return ColorSpace::Rgba;
    case MakeFourCC('A', 'R', '2', '4'):  // V4L2 ARGB32 (BGRA in memory)
      return ColorSpace::Bgra;
    case MakeFourCC('X', 'B', '2', '4'):  // V4L2 XBGR32 (RGBX in memory)
      return ColorSpace::Rgbx;
    case MakeFourCC('X', 'R', '2', '4'):  // V4L2 XRGB32 (BGRX in memory)
      return ColorSpace::Bgrx;

    // Grayscale
    case MakeFourCC('G', 'R', 'E', 'Y'):
    case MakeFourCC('Y', '8', ' ', ' '):
      return ColorSpace::Grey;

    // YUV 4:2:2 packed
    case MakeFourCC('Y', 'U', 'Y', 'V'):
    case MakeFourCC('Y', 'U', 'Y', '2'):
      return ColorSpace::Yuyv;
    case MakeFourCC('U', 'Y', 'V', 'Y'):
      return ColorSpace::Uyvy;
    case MakeFourCC('Y', 'V', 'Y', 'U'):
      return ColorSpace::Yvyu;
    case MakeFourCC('V', 'Y', 'U', 'Y'):
      return ColorSpace::Vyuy;

    // YUV 4:2:2 planar
    case MakeFourCC('Y', 'V', '1', '6'):
      return ColorSpace::Yv16;
    case MakeFourCC('4', '2', '2', 'P'):
      return ColorSpace::I422;

    // YUV 4:2:0 semi-planar
    case MakeFourCC('N', 'V', '1', '2'):
      return ColorSpace::Nv12;
    case MakeFourCC('N', 'V', '2', '1'):
      return ColorSpace::Nv21;

    // YUV 4:2:0 planar
    case MakeFourCC('Y', 'V', '1', '2'):
      return ColorSpace::Yv12;
    case MakeFourCC('Y', 'U', '1', '2'):
    case MakeFourCC('I', '4', '2', '0'):
      return ColorSpace::I420;

    // YUV 4:4:4
    case MakeFourCC('Y', 'U', 'V', '4'):
    case MakeFourCC('Y', '4', '4', '4'):
      return ColorSpace::Yuv444;

    // Bayer
    case MakeFourCC('R', 'G', 'G', 'B'):
      return ColorSpace::Rggb;
    case MakeFourCC('B', 'G', 'G', 'R'):
      return ColorSpace::Bggr;
    case MakeFourCC('G', 'R', 'B', 'G'):
      return ColorSpace::Grbg;
    case MakeFourCC('G', 'B', 'R', 'G'):
      return ColorSpace::Gbrg;

    default:
      return ColorSpace::Rgb;
  }
}

const char* ColorSpaceToString(ColorSpace cs) {
  switch (cs) {
    // RGB/BGR
    case ColorSpace::Rgb:
      return "rgb";
    case ColorSpace::Bgr:
      return "bgr";
    case ColorSpace::Rgba:
      return "rgba";
    case ColorSpace::Bgra:
      return "bgra";
    case ColorSpace::Argb:
      return "argb";
    case ColorSpace::Abgr:
      return "abgr";
    case ColorSpace::Rgbx:
      return "rgbx";
    case ColorSpace::Bgrx:
      return "bgrx";
    case ColorSpace::Xrgb:
      return "xrgb";
    case ColorSpace::Xbgr:
      return "xbgr";
    case ColorSpace::Rgb565:
      return "rgb565";
    case ColorSpace::Bgr565:
      return "bgr565";

    // Grayscale
    case ColorSpace::Grey:
      return "grey";

    // YUV 4:2:2 packed
    case ColorSpace::Yuyv:
      return "yuyv";
    case ColorSpace::Uyvy:
      return "uyvy";
    case ColorSpace::Yvyu:
      return "yvyu";
    case ColorSpace::Vyuy:
      return "vyuy";

    // YUV 4:2:2 planar
    case ColorSpace::Yv16:
      return "yv16";
    case ColorSpace::I422:
      return "i422";

    // YUV 4:2:0 semi-planar
    case ColorSpace::Nv12:
      return "nv12";
    case ColorSpace::Nv21:
      return "nv21";

    // YUV 4:2:0 planar
    case ColorSpace::Yv12:
      return "yv12";
    case ColorSpace::I420:
      return "i420";

    // YUV 4:4:4
    case ColorSpace::Yuv444:
      return "yuv444";

    // Bayer
    case ColorSpace::Rggb:
      return "rggb";
    case ColorSpace::Bggr:
      return "bggr";
    case ColorSpace::Grbg:
      return "grbg";
    case ColorSpace::Gbrg:
      return "gbrg";

    default:
      return "unknown";
  }
}

ColorSpace ColorSpaceFromString(const char* str) {
  if (!str) return ColorSpace::Rgb;

  // RGB/BGR
  if (StrCaseEqual(str, "rgb")) return ColorSpace::Rgb;
  if (StrCaseEqual(str, "bgr")) return ColorSpace::Bgr;
  if (StrCaseEqual(str, "rgba")) return ColorSpace::Rgba;
  if (StrCaseEqual(str, "bgra")) return ColorSpace::Bgra;
  if (StrCaseEqual(str, "argb")) return ColorSpace::Argb;
  if (StrCaseEqual(str, "abgr")) return ColorSpace::Abgr;
  if (StrCaseEqual(str, "rgbx")) return ColorSpace::Rgbx;
  if (StrCaseEqual(str, "bgrx")) return ColorSpace::Bgrx;
  if (StrCaseEqual(str, "xrgb")) return ColorSpace::Xrgb;
  if (StrCaseEqual(str, "xbgr")) return ColorSpace::Xbgr;
  if (StrCaseEqual(str, "rgb565") || StrCaseEqual(str, "rgb16")) {
    return ColorSpace::Rgb565;
  }
  if (StrCaseEqual(str, "bgr565") || StrCaseEqual(str, "bgr16")) {
    return ColorSpace::Bgr565;
  }

  // Grayscale
  if (StrCaseEqual(str, "grey") || StrCaseEqual(str, "gray") ||
      StrCaseEqual(str, "gray8")) {
    return ColorSpace::Grey;
  }

  // YUV 4:2:2 packed
  if (StrCaseEqual(str, "yuyv") || StrCaseEqual(str, "yuy2")) {
    return ColorSpace::Yuyv;
  }
  if (StrCaseEqual(str, "uyvy")) return ColorSpace::Uyvy;
  if (StrCaseEqual(str, "yvyu")) return ColorSpace::Yvyu;
  if (StrCaseEqual(str, "vyuy")) return ColorSpace::Vyuy;

  // YUV 4:2:2 planar
  if (StrCaseEqual(str, "yv16")) return ColorSpace::Yv16;
  if (StrCaseEqual(str, "i422") || StrCaseEqual(str, "y42b")) {
    return ColorSpace::I422;
  }

  // YUV 4:2:0 semi-planar
  if (StrCaseEqual(str, "nv12")) return ColorSpace::Nv12;
  if (StrCaseEqual(str, "nv21")) return ColorSpace::Nv21;

  // YUV 4:2:0 planar
  if (StrCaseEqual(str, "yv12")) return ColorSpace::Yv12;
  if (StrCaseEqual(str, "i420") || StrCaseEqual(str, "iyuv") ||
      StrCaseEqual(str, "yu12")) {
    return ColorSpace::I420;
  }

  // YUV 4:4:4
  if (StrCaseEqual(str, "yuv444") || StrCaseEqual(str, "y444")) {
    return ColorSpace::Yuv444;
  }

  // Bayer
  if (StrCaseEqual(str, "rggb")) return ColorSpace::Rggb;
  if (StrCaseEqual(str, "bggr")) return ColorSpace::Bggr;
  if (StrCaseEqual(str, "grbg")) return ColorSpace::Grbg;
  if (StrCaseEqual(str, "gbrg")) return ColorSpace::Gbrg;

  // Also accept FourCC codes as strings
  return FromFourCC(str);
}

}  // namespace camera_adaptor
}  // namespace edgefirst
