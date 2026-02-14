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

#include "image_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

// stb_image implementation (header-only library)
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace camera_adaptor_test {

Image LoadImage(const std::string& path, int desired_channels) {
  Image img;

  int width, height, channels;
  uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels,
                            desired_channels);

  if (!data) {
    std::cerr << "Failed to load image: " << path << std::endl;
    std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
    return img;
  }

  img.width = width;
  img.height = height;
  img.channels = (desired_channels > 0) ? desired_channels : channels;
  img.data.assign(data, data + img.GetByteSize());

  stbi_image_free(data);
  return img;
}

bool SaveImage(const std::string& path, const Image& image) {
  if (!image.IsValid()) {
    return false;
  }

  int result = stbi_write_png(path.c_str(), image.width, image.height,
                               image.channels, image.data.data(),
                               image.width * image.channels);
  return result != 0;
}

Image ResizeImage(const Image& src, int target_width, int target_height,
                  bool letterbox, uint32_t pad_color) {
  Image dst;

  if (!src.IsValid() || target_width <= 0 || target_height <= 0) {
    return dst;
  }

  dst.width = target_width;
  dst.height = target_height;
  dst.channels = src.channels;
  dst.data.resize(dst.GetByteSize());

  // Extract padding color components
  uint8_t pad_r = (pad_color >> 16) & 0xFF;
  uint8_t pad_g = (pad_color >> 8) & 0xFF;
  uint8_t pad_b = pad_color & 0xFF;
  uint8_t pad_a = 0xFF;

  if (letterbox) {
    // Calculate scale to preserve aspect ratio
    float scale_x = static_cast<float>(target_width) / src.width;
    float scale_y = static_cast<float>(target_height) / src.height;
    float scale = std::min(scale_x, scale_y);

    int scaled_width = static_cast<int>(src.width * scale);
    int scaled_height = static_cast<int>(src.height * scale);

    int offset_x = (target_width - scaled_width) / 2;
    int offset_y = (target_height - scaled_height) / 2;

    // Fill with padding color
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        uint8_t* pixel = dst.data.data() + (y * target_width + x) * dst.channels;
        pixel[0] = pad_r;
        if (dst.channels > 1) pixel[1] = pad_g;
        if (dst.channels > 2) pixel[2] = pad_b;
        if (dst.channels > 3) pixel[3] = pad_a;
      }
    }

    // Bilinear interpolation for scaled region
    for (int y = 0; y < scaled_height; ++y) {
      for (int x = 0; x < scaled_width; ++x) {
        float src_x = x / scale;
        float src_y = y / scale;

        int x0 = static_cast<int>(src_x);
        int y0 = static_cast<int>(src_y);
        int x1 = std::min(x0 + 1, src.width - 1);
        int y1 = std::min(y0 + 1, src.height - 1);

        float fx = src_x - x0;
        float fy = src_y - y0;

        uint8_t* dst_pixel = dst.data.data() +
            ((y + offset_y) * target_width + (x + offset_x)) * dst.channels;

        for (int c = 0; c < src.channels; ++c) {
          float v00 = src.data[(y0 * src.width + x0) * src.channels + c];
          float v10 = src.data[(y0 * src.width + x1) * src.channels + c];
          float v01 = src.data[(y1 * src.width + x0) * src.channels + c];
          float v11 = src.data[(y1 * src.width + x1) * src.channels + c];

          float value = v00 * (1 - fx) * (1 - fy) +
                        v10 * fx * (1 - fy) +
                        v01 * (1 - fx) * fy +
                        v11 * fx * fy;

          dst_pixel[c] = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
        }
      }
    }
  } else {
    // Simple bilinear resize (stretch to fit)
    float scale_x = static_cast<float>(src.width) / target_width;
    float scale_y = static_cast<float>(src.height) / target_height;

    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        float src_x = x * scale_x;
        float src_y = y * scale_y;

        int x0 = static_cast<int>(src_x);
        int y0 = static_cast<int>(src_y);
        int x1 = std::min(x0 + 1, src.width - 1);
        int y1 = std::min(y0 + 1, src.height - 1);

        float fx = src_x - x0;
        float fy = src_y - y0;

        uint8_t* dst_pixel = dst.data.data() + (y * target_width + x) * dst.channels;

        for (int c = 0; c < src.channels; ++c) {
          float v00 = src.data[(y0 * src.width + x0) * src.channels + c];
          float v10 = src.data[(y0 * src.width + x1) * src.channels + c];
          float v01 = src.data[(y1 * src.width + x0) * src.channels + c];
          float v11 = src.data[(y1 * src.width + x1) * src.channels + c];

          float value = v00 * (1 - fx) * (1 - fy) +
                        v10 * fx * (1 - fy) +
                        v01 * (1 - fx) * fy +
                        v11 * fx * fy;

          dst_pixel[c] = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
        }
      }
    }
  }

  return dst;
}

Image ConvertToRGBA(const Image& src) {
  Image dst;

  if (!src.IsValid()) {
    return dst;
  }

  dst.width = src.width;
  dst.height = src.height;
  dst.channels = 4;
  dst.data.resize(dst.GetByteSize());

  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) {
      const uint8_t* src_pixel = src.data.data() + (y * src.width + x) * src.channels;
      uint8_t* dst_pixel = dst.data.data() + (y * src.width + x) * 4;

      if (src.channels == 1) {
        // Grey -> RGBA
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[0];
        dst_pixel[2] = src_pixel[0];
        dst_pixel[3] = 255;
      } else if (src.channels == 3) {
        // RGB -> RGBA
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[1];
        dst_pixel[2] = src_pixel[2];
        dst_pixel[3] = 255;
      } else if (src.channels == 4) {
        // RGBA -> RGBA (copy)
        std::memcpy(dst_pixel, src_pixel, 4);
      }
    }
  }

  return dst;
}

Image ConvertToRGB(const Image& src) {
  Image dst;

  if (!src.IsValid()) {
    return dst;
  }

  dst.width = src.width;
  dst.height = src.height;
  dst.channels = 3;
  dst.data.resize(dst.GetByteSize());

  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) {
      const uint8_t* src_pixel = src.data.data() + (y * src.width + x) * src.channels;
      uint8_t* dst_pixel = dst.data.data() + (y * src.width + x) * 3;

      if (src.channels == 1) {
        // Grey -> RGB
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[0];
        dst_pixel[2] = src_pixel[0];
      } else if (src.channels >= 3) {
        // RGB/RGBA -> RGB
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[1];
        dst_pixel[2] = src_pixel[2];
      }
    }
  }

  return dst;
}

std::vector<std::string> LoadLabels(const std::string& path) {
  std::vector<std::string> labels;

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open labels file: " << path << std::endl;
    return labels;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Remove trailing whitespace
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                             line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }
    labels.push_back(line);
  }

  return labels;
}

}  // namespace camera_adaptor_test
