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

#include "config.h"

namespace edgefirst {
namespace camera_adaptor {

CameraAdaptorConfig CreateConfigFromFourCC(const char* fourcc) {
  CameraAdaptorConfig config;
  config.adaptor = FromFourCC(fourcc);
  config.model_format = ColorSpace::Rgb;

  // Set is_planar based on the color space
  config.is_planar = IsPlanar(config.adaptor);

  return config;
}

CameraAdaptorConfig CreateConfigFromFourCC(uint32_t fourcc) {
  CameraAdaptorConfig config;
  config.adaptor = FromFourCC(fourcc);
  config.model_format = ColorSpace::Rgb;

  // Set is_planar based on the color space
  config.is_planar = IsPlanar(config.adaptor);

  return config;
}

CameraAdaptorConfig CreateConfigFromString(const char* format) {
  CameraAdaptorConfig config;
  config.adaptor = ColorSpaceFromString(format);
  config.model_format = ColorSpace::Rgb;

  // Set is_planar based on the color space
  config.is_planar = IsPlanar(config.adaptor);

  return config;
}

}  // namespace camera_adaptor
}  // namespace edgefirst
