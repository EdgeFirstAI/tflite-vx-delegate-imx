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

#ifndef CAMERA_ADAPTOR_TEST_G2D_HELPER_H_
#define CAMERA_ADAPTOR_TEST_G2D_HELPER_H_

#include <cstdint>
#include <cstddef>

// G2D format constants (matching g2d.h when available)
#ifdef HAVE_G2D
#include <g2d.h>
#else
// Stub constants for compilation when G2D is not available
enum {
  G2D_RGB565 = 0,
  G2D_RGBA8888 = 1,
  G2D_RGBX8888 = 2,
  G2D_BGRA8888 = 3,
  G2D_BGRX8888 = 4,
  G2D_BGR565 = 5,
  G2D_ARGB8888 = 6,
  G2D_ABGR8888 = 7,
  G2D_XRGB8888 = 8,
  G2D_XBGR8888 = 9,
  G2D_RGB888 = 10,
  G2D_BGR888 = 11,
  G2D_YUYV = 20,
  G2D_UYVY = 21,
  G2D_NV12 = 22,
  G2D_NV21 = 23,
};
#endif

namespace camera_adaptor_test {

/**
 * G2D buffer descriptor.
 */
struct G2DBuffer {
  int dmabuf_fd = -1;
  void* virt_addr = nullptr;
  void* phys_addr = nullptr;  // G2D physical address handle
  size_t size = 0;
  int width = 0;
  int height = 0;
  uint32_t format = 0;  // G2D format (e.g., G2D_RGBA8888)
};

/**
 * Letterbox region info - describes where the scaled image goes within the
 * destination buffer, and the padding regions.
 */
struct LetterboxInfo {
  int offset_x = 0;        // Left padding (pixels)
  int offset_y = 0;        // Top padding (pixels)
  int scaled_width = 0;    // Scaled image width
  int scaled_height = 0;   // Scaled image height
};

/**
 * Persistent G2D context for amortizing per-call overhead across iterations.
 * Holds a reusable G2D handle, pre-allocated source buffer, and cached
 * physical address for a DMA-BUF destination.
 */
struct G2DContext {
  void* handle = nullptr;           // g2d_open handle (reused across calls)
  G2DBuffer src_buf = {};           // Pre-allocated source buffer
  unsigned long dmabuf_phys = 0;    // Cached physical address of dest DMA-BUF
  int dmabuf_fd = -1;               // DMA-BUF fd (for phys addr association)
};

#ifdef HAVE_G2D

/**
 * Check if G2D is available on this system.
 */
bool G2DIsAvailable();

/**
 * Allocate a G2D-compatible buffer backed by DMA-BUF.
 *
 * @param buf Buffer descriptor to fill
 * @param width Buffer width in pixels
 * @param height Buffer height in pixels
 * @param format G2D pixel format
 * @return true on success
 */
bool G2DAllocBuffer(G2DBuffer* buf, int width, int height, uint32_t format,
                    bool cacheable = true);

/**
 * Free a G2D buffer.
 */
void G2DFreeBuffer(G2DBuffer* buf);

/**
 * Resize an image using G2D hardware acceleration.
 *
 * @param src_data Source image data (RGB or RGBA)
 * @param src_width Source width
 * @param src_height Source height
 * @param src_channels Source channel count (3=RGB, 4=RGBA)
 * @param dst Pre-allocated destination G2D buffer
 * @param letterbox If true, preserve aspect ratio with padding
 * @param letterbox_color Padding color as packed RGB (0xRRGGBB)
 * @param prefilled If true, skip filling (buffer was pre-filled at init)
 * @return true on success
 */
bool G2DResize(const uint8_t* src_data, int src_width, int src_height,
               int src_channels, G2DBuffer* dst, bool letterbox = false,
               uint32_t letterbox_color = 0x808080, bool prefilled = false);

/**
 * Blit (copy) from one G2D buffer to another with optional scaling.
 *
 * @param src Source buffer
 * @param dst Destination buffer
 * @param letterbox If true, preserve aspect ratio
 * @param letterbox_color Padding color
 * @param prefilled If true, skip filling (buffer was pre-filled at init)
 * @return true on success
 */
bool G2DBlit(const G2DBuffer* src, G2DBuffer* dst, bool letterbox = false,
             uint32_t letterbox_color = 0x808080, bool prefilled = false);

/**
 * Calculate letterbox dimensions for a given source and destination size.
 */
LetterboxInfo CalculateLetterbox(int src_width, int src_height,
                                  int dst_width, int dst_height);

/**
 * Pre-fill a buffer with a solid color (for letterbox padding).
 * Call this ONCE at initialization, then only blit the image region each frame.
 *
 * @param buf Buffer to fill
 * @param color Fill color as packed RGB (0xRRGGBB)
 * @return true on success
 */
bool G2DFillBuffer(G2DBuffer* buf, uint32_t color);

/**
 * Pre-fill a DMA-BUF with a solid color (for letterbox padding).
 * Call this ONCE at initialization, then only blit the image region each frame.
 *
 * @param virt_addr Virtual address of the DMA-BUF
 * @param width Buffer width in pixels
 * @param height Buffer height in pixels
 * @param format G2D pixel format
 * @param color Fill color as packed RGB (0xRRGGBB)
 * @return true on success
 */
bool G2DFillDmaBuf(void* virt_addr, int width, int height, uint32_t format,
                   uint32_t color);

/**
 * Wrap an external DMA-BUF fd for use as a G2D buffer.
 * This enables true zero-copy: G2D outputs directly to a buffer that
 * the NPU can read without any CPU memcpy.
 *
 * @param buf Buffer descriptor to fill
 * @param dmabuf_fd External DMA-BUF file descriptor
 * @param virt_addr Virtual address mapping of the DMA-BUF (from mmap)
 * @param width Buffer width in pixels
 * @param height Buffer height in pixels
 * @param format G2D pixel format
 * @return true on success
 */
bool G2DWrapDmaBuf(G2DBuffer* buf, int dmabuf_fd, void* virt_addr,
                   int width, int height, uint32_t format);

/**
 * Resize an image using G2D, outputting directly to an external DMA-BUF.
 * True zero-copy path: source -> G2D -> DMA-BUF -> NPU (no CPU memcpy).
 *
 * @param src_data Source image data (RGB or RGBA)
 * @param src_width Source width
 * @param src_height Source height
 * @param src_channels Source channel count (3=RGB, 4=RGBA)
 * @param dmabuf_fd Destination DMA-BUF file descriptor
 * @param dmabuf_virt Virtual address of the DMA-BUF
 * @param dst_width Destination width
 * @param dst_height Destination height
 * @param dst_format G2D destination format
 * @param letterbox If true, preserve aspect ratio with padding
 * @param letterbox_color Padding color as packed RGB (0xRRGGBB)
 * @param prefilled If true, skip filling (buffer was pre-filled at init)
 * @return true on success
 */
bool G2DResizeToDmaBuf(const uint8_t* src_data, int src_width, int src_height,
                       int src_channels, int dmabuf_fd, void* dmabuf_virt,
                       int dst_width, int dst_height, uint32_t dst_format,
                       bool letterbox = false, uint32_t letterbox_color = 0x808080,
                       bool prefilled = false);

/**
 * Open a persistent G2D handle.
 */
bool G2DContextOpen(G2DContext* ctx);

/**
 * Close the G2D handle. Does NOT free src_buf.
 */
void G2DContextClose(G2DContext* ctx);

/**
 * Pre-allocate the source buffer in the context.
 */
bool G2DContextAllocSource(G2DContext* ctx, int w, int h, uint32_t fmt);

/**
 * Free the source buffer in the context.
 */
void G2DContextFreeSource(G2DContext* ctx);

/**
 * Cache the physical address of a DMA-BUF fd in the context.
 */
bool G2DContextCachePhysAddr(G2DContext* ctx, int dmabuf_fd);

/**
 * Resize using a persistent context (reuses handle + source buffer).
 * The source buffer must have been pre-allocated with G2DContextAllocSource.
 */
bool G2DResizeCtx(G2DContext* ctx, const uint8_t* src_data,
                  int src_w, int src_h, int src_ch,
                  G2DBuffer* dst, bool letterbox = false,
                  uint32_t letterbox_color = 0x808080,
                  bool prefilled = false);

/**
 * Resize to DMA-BUF using a persistent context (reuses handle, source buffer,
 * and cached physical address).
 */
bool G2DResizeToDmaBufCtx(G2DContext* ctx, const uint8_t* src_data,
                           int src_w, int src_h, int src_ch,
                           void* dmabuf_virt, int dst_w, int dst_h,
                           uint32_t dst_fmt, bool letterbox = false,
                           uint32_t letterbox_color = 0x808080,
                           bool prefilled = false);

#else  // !HAVE_G2D

// Stub implementations when G2D is not available.
// These are called unconditionally in cleanup paths and availability checks.
inline bool G2DIsAvailable() { return false; }
inline void G2DFreeBuffer(G2DBuffer*) {}
inline void G2DContextFreeSource(G2DContext*) {}
inline void G2DContextClose(G2DContext*) {}

#endif  // HAVE_G2D

}  // namespace camera_adaptor_test

#endif  // CAMERA_ADAPTOR_TEST_G2D_HELPER_H_
