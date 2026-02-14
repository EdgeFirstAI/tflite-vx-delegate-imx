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

#ifndef CAMERA_ADAPTOR_TEST_IMAGE_LOADER_H_
#define CAMERA_ADAPTOR_TEST_IMAGE_LOADER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace camera_adaptor_test {

/**
 * Simple image container.
 */
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;  // 1=grey, 3=RGB, 4=RGBA
  std::vector<uint8_t> data;

  size_t GetByteSize() const {
    return static_cast<size_t>(width) * height * channels;
  }

  bool IsValid() const {
    return width > 0 && height > 0 && channels > 0 && !data.empty();
  }
};

/**
 * Load an image from file (JPEG, PNG, BMP, etc.).
 *
 * Uses stb_image internally for format support.
 *
 * @param path Path to image file
 * @param desired_channels Desired output channels (0=keep original, 3=RGB, 4=RGBA)
 * @return Loaded image, or invalid image on failure
 */
Image LoadImage(const std::string& path, int desired_channels = 0);

/**
 * Save an image to file (PNG format).
 *
 * @param path Output path
 * @param image Image to save
 * @return true on success
 */
bool SaveImage(const std::string& path, const Image& image);

/**
 * Resize image using software bilinear interpolation.
 *
 * This is the fallback when G2D is not available.
 *
 * @param src Source image
 * @param target_width Target width
 * @param target_height Target height
 * @param letterbox If true, preserve aspect ratio with padding
 * @param pad_color Padding color (R, G, B for 3-channel or R, G, B, A for 4-channel)
 * @return Resized image
 */
Image ResizeImage(const Image& src, int target_width, int target_height,
                  bool letterbox = false, uint32_t pad_color = 0x808080);

/**
 * Convert image to RGBA format.
 *
 * @param src Source image (any channel count)
 * @return RGBA image (4 channels)
 */
Image ConvertToRGBA(const Image& src);

/**
 * Convert image to RGB format.
 *
 * @param src Source image (any channel count)
 * @return RGB image (3 channels)
 */
Image ConvertToRGB(const Image& src);

/**
 * Load labels from text file (one label per line).
 *
 * @param path Path to labels file
 * @return Vector of labels, empty on failure
 */
std::vector<std::string> LoadLabels(const std::string& path);

}  // namespace camera_adaptor_test

#endif  // CAMERA_ADAPTOR_TEST_IMAGE_LOADER_H_
