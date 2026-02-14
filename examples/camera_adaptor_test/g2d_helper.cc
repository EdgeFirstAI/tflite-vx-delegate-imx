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

#include "g2d_helper.h"

#include <algorithm>

#ifdef HAVE_G2D

#include <g2d.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <sys/ioctl.h>

// DMA-BUF physical address ioctl (NXP i.MX extension)
#ifndef DMA_BUF_BASE
#define DMA_BUF_BASE 'b'
#endif
#ifndef DMA_BUF_IOCTL_PHYS
struct dma_buf_phys {
  unsigned long phys;
};
#define DMA_BUF_IOCTL_PHYS _IOW(DMA_BUF_BASE, 10, struct dma_buf_phys)
#endif

namespace camera_adaptor_test {

/**
 * Get the physical address of a DMA-BUF using i.MX-specific ioctl.
 * Returns 0 on failure.
 */
static unsigned long GetDmaBufPhysAddr(int dmabuf_fd) {
  struct dma_buf_phys phys = {};
  if (ioctl(dmabuf_fd, DMA_BUF_IOCTL_PHYS, &phys) < 0) {
    std::cerr << "G2D: DMA_BUF_IOCTL_PHYS failed: " << strerror(errno) << std::endl;
    return 0;
  }
  return phys.phys;
}

// Get bytes per pixel for G2D format
static int GetBytesPerPixel(uint32_t format) {
  switch (format) {
    case G2D_RGBA8888:
    case G2D_BGRA8888:
    case G2D_ARGB8888:
    case G2D_ABGR8888:
    case G2D_RGBX8888:
    case G2D_XRGB8888:
      return 4;
    case G2D_RGB888:
    case G2D_BGR888:
      return 3;
    case G2D_RGB565:
    case G2D_YUYV:
    case G2D_UYVY:
      return 2;
    default:
      return 4;
  }
}

bool G2DIsAvailable() {
  void* handle = nullptr;
  if (g2d_open(&handle) == 0 && handle) {
    g2d_close(handle);
    return true;
  }
  return false;
}

bool G2DAllocBuffer(G2DBuffer* buf, int width, int height, uint32_t format,
                    bool cacheable) {
  if (!buf || width <= 0 || height <= 0) {
    return false;
  }

  int bpp = GetBytesPerPixel(format);
  size_t size = static_cast<size_t>(width) * height * bpp;

  struct g2d_buf* g2d_buffer = g2d_alloc(size, cacheable ? 1 : 0);
  if (!g2d_buffer) {
    std::cerr << "G2D: g2d_alloc failed for " << size << " bytes" << std::endl;
    return false;
  }

  buf->phys_addr = g2d_buffer;
  buf->virt_addr = g2d_buffer->buf_vaddr;
  buf->size = size;
  buf->width = width;
  buf->height = height;
  buf->format = format;

  // Note: G2D buffers from g2d_alloc don't expose a dmabuf_fd directly.
  // For DMA-BUF integration, we would need to use a different allocation
  // path (e.g., ION or dma_heap with g2d_buf_from_fd).
  buf->dmabuf_fd = -1;

  return true;
}

void G2DFreeBuffer(G2DBuffer* buf) {
  if (!buf) return;

  if (buf->phys_addr) {
    g2d_free(static_cast<struct g2d_buf*>(buf->phys_addr));
  }

  buf->phys_addr = nullptr;
  buf->virt_addr = nullptr;
  buf->dmabuf_fd = -1;
  buf->size = 0;
}

LetterboxInfo CalculateLetterbox(int src_width, int src_height,
                                  int dst_width, int dst_height) {
  LetterboxInfo info;
  float scale_x = static_cast<float>(dst_width) / src_width;
  float scale_y = static_cast<float>(dst_height) / src_height;
  float scale = std::min(scale_x, scale_y);

  info.scaled_width = static_cast<int>(src_width * scale);
  info.scaled_height = static_cast<int>(src_height * scale);
  info.offset_x = (dst_width - info.scaled_width) / 2;
  info.offset_y = (dst_height - info.scaled_height) / 2;
  return info;
}

bool G2DFillBuffer(G2DBuffer* buf, uint32_t color) {
  if (!buf || !buf->virt_addr) {
    return false;
  }
  return G2DFillDmaBuf(buf->virt_addr, buf->width, buf->height, buf->format, color);
}

bool G2DFillDmaBuf(void* virt_addr, int width, int height, uint32_t format,
                   uint32_t color) {
  if (!virt_addr || width <= 0 || height <= 0) {
    return false;
  }

  uint8_t* pixels = static_cast<uint8_t*>(virt_addr);
  int bpp = GetBytesPerPixel(format);
  uint8_t r = (color >> 16) & 0xFF;
  uint8_t g = (color >> 8) & 0xFF;
  uint8_t b = color & 0xFF;

  int num_pixels = width * height;

  if (bpp == 4) {
    // RGBA/BGRA/etc - 4 bytes per pixel
    for (int i = 0; i < num_pixels; ++i) {
      pixels[i * 4 + 0] = r;
      pixels[i * 4 + 1] = g;
      pixels[i * 4 + 2] = b;
      pixels[i * 4 + 3] = 0xFF;  // Alpha
    }
  } else if (bpp == 3) {
    // RGB/BGR - 3 bytes per pixel
    for (int i = 0; i < num_pixels; ++i) {
      pixels[i * 3 + 0] = r;
      pixels[i * 3 + 1] = g;
      pixels[i * 3 + 2] = b;
    }
  }

  return true;
}

bool G2DResize(const uint8_t* src_data, int src_width, int src_height,
               int src_channels, G2DBuffer* dst, bool letterbox,
               uint32_t letterbox_color, bool prefilled) {
  if (!src_data || !dst || !dst->virt_addr) {
    return false;
  }

  // Allocate temporary source buffer
  G2DBuffer src_buf;
  uint32_t src_format = (src_channels == 4) ? G2D_RGBA8888 : G2D_RGB888;
  if (!G2DAllocBuffer(&src_buf, src_width, src_height, src_format)) {
    std::cerr << "G2D: Failed to allocate source buffer" << std::endl;
    return false;
  }

  // Copy source data to G2D buffer
  std::memcpy(src_buf.virt_addr, src_data, src_buf.size);
  g2d_cache_op(static_cast<struct g2d_buf*>(src_buf.phys_addr), G2D_CACHE_FLUSH);

  // Perform blit
  bool result = G2DBlit(&src_buf, dst, letterbox, letterbox_color, prefilled);

  // Cleanup source buffer
  G2DFreeBuffer(&src_buf);

  return result;
}

bool G2DBlit(const G2DBuffer* src, G2DBuffer* dst, bool letterbox,
             uint32_t letterbox_color, bool prefilled) {
  if (!src || !dst || !src->phys_addr || !dst->phys_addr) {
    return false;
  }

  void* handle = nullptr;
  if (g2d_open(&handle) != 0 || !handle) {
    std::cerr << "G2D: Failed to open G2D device" << std::endl;
    return false;
  }

  struct g2d_surface src_surf = {};
  struct g2d_surface dst_surf = {};

  // Setup source surface
  src_surf.format = static_cast<g2d_format>(src->format);
  src_surf.planes[0] = static_cast<struct g2d_buf*>(
      const_cast<void*>(src->phys_addr))->buf_paddr;
  src_surf.left = 0;
  src_surf.top = 0;
  src_surf.right = src->width;
  src_surf.bottom = src->height;
  src_surf.stride = src->width;
  src_surf.width = src->width;
  src_surf.height = src->height;
  src_surf.blendfunc = G2D_ONE;
  src_surf.global_alpha = 0xFF;

  // Setup destination surface
  dst_surf.format = static_cast<g2d_format>(dst->format);
  dst_surf.planes[0] = static_cast<struct g2d_buf*>(dst->phys_addr)->buf_paddr;
  dst_surf.stride = dst->width;
  dst_surf.width = dst->width;
  dst_surf.height = dst->height;
  dst_surf.blendfunc = G2D_ONE;
  dst_surf.global_alpha = 0xFF;

  // Track if we need CPU-based letterbox fill (for formats G2D can't clear)
  bool needs_cpu_fill = false;
  int letterbox_offset_x = 0;
  int letterbox_offset_y = 0;
  int letterbox_scaled_width = dst->width;
  int letterbox_scaled_height = dst->height;

  if (letterbox) {
    // Calculate letterbox region first
    float scale_x = static_cast<float>(dst->width) / src->width;
    float scale_y = static_cast<float>(dst->height) / src->height;
    float scale = std::min(scale_x, scale_y);

    letterbox_scaled_width = static_cast<int>(src->width * scale);
    letterbox_scaled_height = static_cast<int>(src->height * scale);
    letterbox_offset_x = (dst->width - letterbox_scaled_width) / 2;
    letterbox_offset_y = (dst->height - letterbox_scaled_height) / 2;

    if (!prefilled) {
      // G2D clear doesn't support RGB888/BGR888 format (format 10/11)
      // For these formats, we'll do CPU-based fill after the blit
      bool is_rgb_format = (dst->format == G2D_RGB888 || dst->format == G2D_BGR888);

      if (!is_rgb_format) {
        // Clear destination with letterbox color using G2D
        dst_surf.clrcolor = (0xFF << 24) |  // Alpha
                            ((letterbox_color >> 16) & 0xFF) |  // R -> B (BGRA)
                            ((letterbox_color >> 8) & 0xFF) << 8 |  // G
                            (letterbox_color & 0xFF) << 16;  // B -> R

        dst_surf.left = 0;
        dst_surf.top = 0;
        dst_surf.right = dst->width;
        dst_surf.bottom = dst->height;

        if (g2d_clear(handle, &dst_surf) != 0) {
          std::cerr << "G2D: g2d_clear failed" << std::endl;
          g2d_close(handle);
          return false;
        }
      } else {
        // Mark for CPU fill after blit
        needs_cpu_fill = true;
      }
    }

    dst_surf.left = letterbox_offset_x;
    dst_surf.top = letterbox_offset_y;
    dst_surf.right = letterbox_offset_x + letterbox_scaled_width;
    dst_surf.bottom = letterbox_offset_y + letterbox_scaled_height;
  } else {
    dst_surf.left = 0;
    dst_surf.top = 0;
    dst_surf.right = dst->width;
    dst_surf.bottom = dst->height;
  }

  // Perform blit with scaling
  if (g2d_blit(handle, &src_surf, &dst_surf) != 0) {
    std::cerr << "G2D: g2d_blit failed" << std::endl;
    g2d_close(handle);
    return false;
  }

  // Wait for G2D to complete
  g2d_finish(handle);

  // Invalidate cache so CPU can read G2D's output correctly.
  // Required because G2D wrote via DMA (bypassing CPU cache) but
  // CPU will read this buffer to copy to the model input tensor.
  g2d_cache_op(static_cast<struct g2d_buf*>(dst->phys_addr), G2D_CACHE_INVALIDATE);

  g2d_close(handle);

  // For RGB format with letterbox, do CPU-based fill for the padding areas
  if (needs_cpu_fill && dst->virt_addr) {
    uint8_t* pixels = static_cast<uint8_t*>(dst->virt_addr);
    int bpp = GetBytesPerPixel(dst->format);
    uint8_t r = (letterbox_color >> 16) & 0xFF;
    uint8_t g = (letterbox_color >> 8) & 0xFF;
    uint8_t b = letterbox_color & 0xFF;

    // Fill top padding
    for (int y = 0; y < letterbox_offset_y; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }

    // Fill bottom padding
    for (int y = letterbox_offset_y + letterbox_scaled_height; y < dst->height; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }

    // Fill left padding
    for (int y = letterbox_offset_y; y < letterbox_offset_y + letterbox_scaled_height; ++y) {
      for (int x = 0; x < letterbox_offset_x; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }

    // Fill right padding
    for (int y = letterbox_offset_y; y < letterbox_offset_y + letterbox_scaled_height; ++y) {
      for (int x = letterbox_offset_x + letterbox_scaled_width; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
  }

  return true;
}

bool G2DWrapDmaBuf(G2DBuffer* buf, int dmabuf_fd, void* virt_addr,
                   int width, int height, uint32_t format) {
  if (!buf || dmabuf_fd < 0 || !virt_addr || width <= 0 || height <= 0) {
    return false;
  }

  // Get physical address of DMA-BUF using i.MX-specific ioctl
  unsigned long phys_addr = GetDmaBufPhysAddr(dmabuf_fd);
  if (phys_addr == 0) {
    std::cerr << "G2D: Failed to get physical address for DMA-BUF fd=" << dmabuf_fd << std::endl;
    return false;
  }

  int bpp = GetBytesPerPixel(format);
  size_t size = static_cast<size_t>(width) * height * bpp;

  // Store physical address directly - G2D surface will use this
  buf->phys_addr = reinterpret_cast<void*>(phys_addr);
  buf->virt_addr = virt_addr;
  buf->dmabuf_fd = dmabuf_fd;
  buf->size = size;
  buf->width = width;
  buf->height = height;
  buf->format = format;

  return true;
}

/**
 * G2D blit using physical address directly (for DMA-BUF destination).
 * This variant uses the physical address from G2DWrapDmaBuf.
 */
static bool G2DBlitToPhys(const G2DBuffer* src, G2DBuffer* dst, bool letterbox,
                          uint32_t letterbox_color, bool prefilled) {
  if (!src || !dst || !src->phys_addr) {
    return false;
  }

  // Check if dst has a g2d_buf (from g2d_alloc) or just a physical address (from DMA-BUF)
  bool dst_is_dmabuf = (dst->dmabuf_fd >= 0);

  void* handle = nullptr;
  if (g2d_open(&handle) != 0 || !handle) {
    std::cerr << "G2D: Failed to open G2D device" << std::endl;
    return false;
  }

  struct g2d_surface src_surf = {};
  struct g2d_surface dst_surf = {};

  // Setup source surface (always from g2d_alloc buffer)
  src_surf.format = static_cast<g2d_format>(src->format);
  src_surf.planes[0] = static_cast<struct g2d_buf*>(
      const_cast<void*>(src->phys_addr))->buf_paddr;
  src_surf.left = 0;
  src_surf.top = 0;
  src_surf.right = src->width;
  src_surf.bottom = src->height;
  src_surf.stride = src->width;
  src_surf.width = src->width;
  src_surf.height = src->height;
  src_surf.blendfunc = G2D_ONE;
  src_surf.global_alpha = 0xFF;

  // Setup destination surface
  dst_surf.format = static_cast<g2d_format>(dst->format);
  if (dst_is_dmabuf) {
    // Destination is DMA-BUF - use physical address directly
    dst_surf.planes[0] = reinterpret_cast<unsigned long>(dst->phys_addr);
  } else {
    // Destination is g2d_alloc buffer
    dst_surf.planes[0] = static_cast<struct g2d_buf*>(dst->phys_addr)->buf_paddr;
  }
  dst_surf.stride = dst->width;
  dst_surf.width = dst->width;
  dst_surf.height = dst->height;
  dst_surf.blendfunc = G2D_ONE;
  dst_surf.global_alpha = 0xFF;

  // Handle letterbox
  int letterbox_offset_x = 0;
  int letterbox_offset_y = 0;
  int letterbox_scaled_width = dst->width;
  int letterbox_scaled_height = dst->height;
  bool needs_cpu_fill = false;

  if (letterbox) {
    float scale_x = static_cast<float>(dst->width) / src->width;
    float scale_y = static_cast<float>(dst->height) / src->height;
    float scale = std::min(scale_x, scale_y);

    letterbox_scaled_width = static_cast<int>(src->width * scale);
    letterbox_scaled_height = static_cast<int>(src->height * scale);
    letterbox_offset_x = (dst->width - letterbox_scaled_width) / 2;
    letterbox_offset_y = (dst->height - letterbox_scaled_height) / 2;

    bool is_rgb_format = (dst->format == G2D_RGB888 || dst->format == G2D_BGR888);

    if (!prefilled) {
      if (!is_rgb_format) {
        dst_surf.clrcolor = (0xFF << 24) |
                            ((letterbox_color >> 16) & 0xFF) |
                            ((letterbox_color >> 8) & 0xFF) << 8 |
                            (letterbox_color & 0xFF) << 16;

        dst_surf.left = 0;
        dst_surf.top = 0;
        dst_surf.right = dst->width;
        dst_surf.bottom = dst->height;

        if (g2d_clear(handle, &dst_surf) != 0) {
          std::cerr << "G2D: g2d_clear failed" << std::endl;
          g2d_close(handle);
          return false;
        }
      } else {
        needs_cpu_fill = true;
      }
    }

    dst_surf.left = letterbox_offset_x;
    dst_surf.top = letterbox_offset_y;
    dst_surf.right = letterbox_offset_x + letterbox_scaled_width;
    dst_surf.bottom = letterbox_offset_y + letterbox_scaled_height;
  } else {
    dst_surf.left = 0;
    dst_surf.top = 0;
    dst_surf.right = dst->width;
    dst_surf.bottom = dst->height;
  }

  // Perform blit with scaling
  if (g2d_blit(handle, &src_surf, &dst_surf) != 0) {
    std::cerr << "G2D: g2d_blit failed" << std::endl;
    g2d_close(handle);
    return false;
  }

  // Wait for G2D to complete
  g2d_finish(handle);
  g2d_close(handle);

  // Note: No cache invalidate needed for DMA-BUF destination.
  // Both G2D and NPU access memory via physical addresses (DMA),
  // bypassing CPU cache entirely. This is the "zero-copy" benefit.

  // For RGB format with letterbox, do CPU-based fill
  if (needs_cpu_fill && dst->virt_addr) {
    uint8_t* pixels = static_cast<uint8_t*>(dst->virt_addr);
    int bpp = GetBytesPerPixel(dst->format);
    uint8_t r = (letterbox_color >> 16) & 0xFF;
    uint8_t g = (letterbox_color >> 8) & 0xFF;
    uint8_t b = letterbox_color & 0xFF;

    for (int y = 0; y < letterbox_offset_y; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y + letterbox_scaled_height; y < dst->height; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y; y < letterbox_offset_y + letterbox_scaled_height; ++y) {
      for (int x = 0; x < letterbox_offset_x; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
      for (int x = letterbox_offset_x + letterbox_scaled_width; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
  }

  return true;
}

bool G2DResizeToDmaBuf(const uint8_t* src_data, int src_width, int src_height,
                       int src_channels, int dmabuf_fd, void* dmabuf_virt,
                       int dst_width, int dst_height, uint32_t dst_format,
                       bool letterbox, uint32_t letterbox_color, bool prefilled) {
  if (!src_data || dmabuf_fd < 0 || !dmabuf_virt) {
    return false;
  }

  // Wrap the DMA-BUF as G2D destination (gets physical address)
  G2DBuffer dst_buf;
  if (!G2DWrapDmaBuf(&dst_buf, dmabuf_fd, dmabuf_virt,
                     dst_width, dst_height, dst_format)) {
    return false;
  }

  // Allocate temporary source buffer (need g2d_alloc for source)
  G2DBuffer src_buf;
  uint32_t src_format = (src_channels == 4) ? G2D_RGBA8888 : G2D_RGB888;
  if (!G2DAllocBuffer(&src_buf, src_width, src_height, src_format)) {
    std::cerr << "G2D: Failed to allocate source buffer" << std::endl;
    return false;
  }

  // Copy source data to G2D buffer
  std::memcpy(src_buf.virt_addr, src_data, src_buf.size);
  g2d_cache_op(static_cast<struct g2d_buf*>(src_buf.phys_addr), G2D_CACHE_FLUSH);

  // Perform blit to the DMA-BUF (true zero-copy for destination)
  bool result = G2DBlitToPhys(&src_buf, &dst_buf, letterbox, letterbox_color, prefilled);

  // Cleanup source buffer only (dst is external DMA-BUF)
  G2DFreeBuffer(&src_buf);

  return result;
}

// ============================================================================
// G2DContext: persistent context for amortizing per-call overhead
// ============================================================================

bool G2DContextOpen(G2DContext* ctx) {
  if (!ctx) return false;
  if (ctx->handle) return true;  // Already open
  if (g2d_open(&ctx->handle) != 0 || !ctx->handle) {
    std::cerr << "G2D: Failed to open G2D device for context" << std::endl;
    ctx->handle = nullptr;
    return false;
  }
  return true;
}

void G2DContextClose(G2DContext* ctx) {
  if (!ctx) return;
  if (ctx->handle) {
    g2d_close(ctx->handle);
    ctx->handle = nullptr;
  }
  ctx->dmabuf_phys = 0;
  ctx->dmabuf_fd = -1;
}

bool G2DContextAllocSource(G2DContext* ctx, int w, int h, uint32_t fmt) {
  if (!ctx) return false;
  return G2DAllocBuffer(&ctx->src_buf, w, h, fmt);
}

void G2DContextFreeSource(G2DContext* ctx) {
  if (!ctx) return;
  G2DFreeBuffer(&ctx->src_buf);
}

bool G2DContextCachePhysAddr(G2DContext* ctx, int dmabuf_fd) {
  if (!ctx || dmabuf_fd < 0) return false;
  unsigned long phys = GetDmaBufPhysAddr(dmabuf_fd);
  if (phys == 0) return false;
  ctx->dmabuf_phys = phys;
  ctx->dmabuf_fd = dmabuf_fd;
  return true;
}

/**
 * Internal blit helper using a pre-opened handle (no g2d_open/close per call).
 */
static bool G2DBlitCtx(void* handle, const G2DBuffer* src, G2DBuffer* dst,
                        bool letterbox, uint32_t letterbox_color,
                        bool prefilled) {
  if (!handle || !src || !dst || !src->phys_addr || !dst->phys_addr) {
    return false;
  }

  struct g2d_surface src_surf = {};
  struct g2d_surface dst_surf = {};

  src_surf.format = static_cast<g2d_format>(src->format);
  src_surf.planes[0] = static_cast<struct g2d_buf*>(
      const_cast<void*>(src->phys_addr))->buf_paddr;
  src_surf.left = 0;
  src_surf.top = 0;
  src_surf.right = src->width;
  src_surf.bottom = src->height;
  src_surf.stride = src->width;
  src_surf.width = src->width;
  src_surf.height = src->height;
  src_surf.blendfunc = G2D_ONE;
  src_surf.global_alpha = 0xFF;

  dst_surf.format = static_cast<g2d_format>(dst->format);
  dst_surf.planes[0] = static_cast<struct g2d_buf*>(dst->phys_addr)->buf_paddr;
  dst_surf.stride = dst->width;
  dst_surf.width = dst->width;
  dst_surf.height = dst->height;
  dst_surf.blendfunc = G2D_ONE;
  dst_surf.global_alpha = 0xFF;

  bool needs_cpu_fill = false;
  int letterbox_offset_x = 0;
  int letterbox_offset_y = 0;
  int letterbox_scaled_width = dst->width;
  int letterbox_scaled_height = dst->height;

  if (letterbox) {
    float scale_x = static_cast<float>(dst->width) / src->width;
    float scale_y = static_cast<float>(dst->height) / src->height;
    float scale = std::min(scale_x, scale_y);

    letterbox_scaled_width = static_cast<int>(src->width * scale);
    letterbox_scaled_height = static_cast<int>(src->height * scale);
    letterbox_offset_x = (dst->width - letterbox_scaled_width) / 2;
    letterbox_offset_y = (dst->height - letterbox_scaled_height) / 2;

    if (!prefilled) {
      bool is_rgb_format = (dst->format == G2D_RGB888 || dst->format == G2D_BGR888);
      if (!is_rgb_format) {
        dst_surf.clrcolor = (0xFF << 24) |
                            ((letterbox_color >> 16) & 0xFF) |
                            ((letterbox_color >> 8) & 0xFF) << 8 |
                            (letterbox_color & 0xFF) << 16;
        dst_surf.left = 0;
        dst_surf.top = 0;
        dst_surf.right = dst->width;
        dst_surf.bottom = dst->height;
        if (g2d_clear(handle, &dst_surf) != 0) {
          std::cerr << "G2D: g2d_clear failed" << std::endl;
          return false;
        }
      } else {
        needs_cpu_fill = true;
      }
    }

    dst_surf.left = letterbox_offset_x;
    dst_surf.top = letterbox_offset_y;
    dst_surf.right = letterbox_offset_x + letterbox_scaled_width;
    dst_surf.bottom = letterbox_offset_y + letterbox_scaled_height;
  } else {
    dst_surf.left = 0;
    dst_surf.top = 0;
    dst_surf.right = dst->width;
    dst_surf.bottom = dst->height;
  }

  if (g2d_blit(handle, &src_surf, &dst_surf) != 0) {
    std::cerr << "G2D: g2d_blit failed" << std::endl;
    return false;
  }

  g2d_finish(handle);
  g2d_cache_op(static_cast<struct g2d_buf*>(dst->phys_addr), G2D_CACHE_INVALIDATE);

  if (needs_cpu_fill && dst->virt_addr) {
    uint8_t* pixels = static_cast<uint8_t*>(dst->virt_addr);
    int bpp = GetBytesPerPixel(dst->format);
    uint8_t r = (letterbox_color >> 16) & 0xFF;
    uint8_t g = (letterbox_color >> 8) & 0xFF;
    uint8_t b = letterbox_color & 0xFF;

    for (int y = 0; y < letterbox_offset_y; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y + letterbox_scaled_height; y < dst->height; ++y) {
      for (int x = 0; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y; y < letterbox_offset_y + letterbox_scaled_height; ++y) {
      for (int x = 0; x < letterbox_offset_x; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
      for (int x = letterbox_offset_x + letterbox_scaled_width; x < dst->width; ++x) {
        uint8_t* p = pixels + (y * dst->width + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
  }

  return true;
}

/**
 * Internal blit to physical address using a pre-opened handle.
 */
static bool G2DBlitToPhysCtx(void* handle, const G2DBuffer* src,
                              unsigned long dst_phys, void* dst_virt,
                              int dst_w, int dst_h, uint32_t dst_fmt,
                              bool letterbox, uint32_t letterbox_color,
                              bool prefilled) {
  if (!handle || !src || !src->phys_addr || dst_phys == 0) {
    return false;
  }

  struct g2d_surface src_surf = {};
  struct g2d_surface dst_surf = {};

  src_surf.format = static_cast<g2d_format>(src->format);
  src_surf.planes[0] = static_cast<struct g2d_buf*>(
      const_cast<void*>(src->phys_addr))->buf_paddr;
  src_surf.left = 0;
  src_surf.top = 0;
  src_surf.right = src->width;
  src_surf.bottom = src->height;
  src_surf.stride = src->width;
  src_surf.width = src->width;
  src_surf.height = src->height;
  src_surf.blendfunc = G2D_ONE;
  src_surf.global_alpha = 0xFF;

  dst_surf.format = static_cast<g2d_format>(dst_fmt);
  dst_surf.planes[0] = dst_phys;
  dst_surf.stride = dst_w;
  dst_surf.width = dst_w;
  dst_surf.height = dst_h;
  dst_surf.blendfunc = G2D_ONE;
  dst_surf.global_alpha = 0xFF;

  int letterbox_offset_x = 0;
  int letterbox_offset_y = 0;
  int letterbox_scaled_width = dst_w;
  int letterbox_scaled_height = dst_h;
  bool needs_cpu_fill = false;

  if (letterbox) {
    float scale_x = static_cast<float>(dst_w) / src->width;
    float scale_y = static_cast<float>(dst_h) / src->height;
    float scale = std::min(scale_x, scale_y);

    letterbox_scaled_width = static_cast<int>(src->width * scale);
    letterbox_scaled_height = static_cast<int>(src->height * scale);
    letterbox_offset_x = (dst_w - letterbox_scaled_width) / 2;
    letterbox_offset_y = (dst_h - letterbox_scaled_height) / 2;

    bool is_rgb_format = (dst_fmt == G2D_RGB888 || dst_fmt == G2D_BGR888);

    if (!prefilled) {
      if (!is_rgb_format) {
        dst_surf.clrcolor = (0xFF << 24) |
                            ((letterbox_color >> 16) & 0xFF) |
                            ((letterbox_color >> 8) & 0xFF) << 8 |
                            (letterbox_color & 0xFF) << 16;
        dst_surf.left = 0;
        dst_surf.top = 0;
        dst_surf.right = dst_w;
        dst_surf.bottom = dst_h;
        if (g2d_clear(handle, &dst_surf) != 0) {
          std::cerr << "G2D: g2d_clear failed" << std::endl;
          return false;
        }
      } else {
        needs_cpu_fill = true;
      }
    }

    dst_surf.left = letterbox_offset_x;
    dst_surf.top = letterbox_offset_y;
    dst_surf.right = letterbox_offset_x + letterbox_scaled_width;
    dst_surf.bottom = letterbox_offset_y + letterbox_scaled_height;
  } else {
    dst_surf.left = 0;
    dst_surf.top = 0;
    dst_surf.right = dst_w;
    dst_surf.bottom = dst_h;
  }

  if (g2d_blit(handle, &src_surf, &dst_surf) != 0) {
    std::cerr << "G2D: g2d_blit failed" << std::endl;
    return false;
  }

  g2d_finish(handle);

  if (needs_cpu_fill && dst_virt) {
    uint8_t* pixels = static_cast<uint8_t*>(dst_virt);
    int bpp = GetBytesPerPixel(dst_fmt);
    uint8_t r = (letterbox_color >> 16) & 0xFF;
    uint8_t g = (letterbox_color >> 8) & 0xFF;
    uint8_t b = letterbox_color & 0xFF;

    for (int y = 0; y < letterbox_offset_y; ++y) {
      for (int x = 0; x < dst_w; ++x) {
        uint8_t* p = pixels + (y * dst_w + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y + letterbox_scaled_height; y < dst_h; ++y) {
      for (int x = 0; x < dst_w; ++x) {
        uint8_t* p = pixels + (y * dst_w + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
    for (int y = letterbox_offset_y; y < letterbox_offset_y + letterbox_scaled_height; ++y) {
      for (int x = 0; x < letterbox_offset_x; ++x) {
        uint8_t* p = pixels + (y * dst_w + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
      for (int x = letterbox_offset_x + letterbox_scaled_width; x < dst_w; ++x) {
        uint8_t* p = pixels + (y * dst_w + x) * bpp;
        p[0] = r; p[1] = g; p[2] = b;
      }
    }
  }

  return true;
}

bool G2DResizeCtx(G2DContext* ctx, const uint8_t* src_data,
                  int src_w, int src_h, int src_ch,
                  G2DBuffer* dst, bool letterbox,
                  uint32_t letterbox_color, bool prefilled) {
  if (!ctx || !ctx->handle || !ctx->src_buf.virt_addr || !src_data || !dst) {
    return false;
  }

  // Copy source data to pre-allocated G2D buffer and flush
  std::memcpy(ctx->src_buf.virt_addr, src_data, ctx->src_buf.size);
  g2d_cache_op(static_cast<struct g2d_buf*>(ctx->src_buf.phys_addr), G2D_CACHE_FLUSH);

  return G2DBlitCtx(ctx->handle, &ctx->src_buf, dst,
                     letterbox, letterbox_color, prefilled);
}

bool G2DResizeToDmaBufCtx(G2DContext* ctx, const uint8_t* src_data,
                           int src_w, int src_h, int src_ch,
                           void* dmabuf_virt, int dst_w, int dst_h,
                           uint32_t dst_fmt, bool letterbox,
                           uint32_t letterbox_color, bool prefilled) {
  if (!ctx || !ctx->handle || !ctx->src_buf.virt_addr ||
      !src_data || ctx->dmabuf_phys == 0) {
    return false;
  }

  // Copy source data to pre-allocated G2D buffer and flush
  std::memcpy(ctx->src_buf.virt_addr, src_data, ctx->src_buf.size);
  g2d_cache_op(static_cast<struct g2d_buf*>(ctx->src_buf.phys_addr), G2D_CACHE_FLUSH);

  return G2DBlitToPhysCtx(ctx->handle, &ctx->src_buf,
                           ctx->dmabuf_phys, dmabuf_virt,
                           dst_w, dst_h, dst_fmt,
                           letterbox, letterbox_color, prefilled);
}

}  // namespace camera_adaptor_test

#else  // !HAVE_G2D

// Stub implementations when G2D is not available

namespace camera_adaptor_test {

bool G2DIsAvailable() {
  return false;
}

bool G2DAllocBuffer(G2DBuffer* buf, int width, int height, uint32_t format,
                    bool cacheable) {
  (void)buf;
  (void)width;
  (void)height;
  (void)format;
  (void)cacheable;
  return false;
}

void G2DFreeBuffer(G2DBuffer* buf) {
  (void)buf;
}

LetterboxInfo CalculateLetterbox(int src_width, int src_height,
                                  int dst_width, int dst_height) {
  LetterboxInfo info;
  float scale_x = static_cast<float>(dst_width) / src_width;
  float scale_y = static_cast<float>(dst_height) / src_height;
  float scale = std::min(scale_x, scale_y);

  info.scaled_width = static_cast<int>(src_width * scale);
  info.scaled_height = static_cast<int>(src_height * scale);
  info.offset_x = (dst_width - info.scaled_width) / 2;
  info.offset_y = (dst_height - info.scaled_height) / 2;
  return info;
}

bool G2DFillBuffer(G2DBuffer* buf, uint32_t color) {
  (void)buf;
  (void)color;
  return false;
}

bool G2DFillDmaBuf(void* virt_addr, int width, int height, uint32_t format,
                   uint32_t color) {
  (void)virt_addr;
  (void)width;
  (void)height;
  (void)format;
  (void)color;
  return false;
}

bool G2DResize(const uint8_t* src_data, int src_width, int src_height,
               int src_channels, G2DBuffer* dst, bool letterbox,
               uint32_t letterbox_color, bool prefilled) {
  (void)src_data;
  (void)src_width;
  (void)src_height;
  (void)src_channels;
  (void)dst;
  (void)letterbox;
  (void)letterbox_color;
  (void)prefilled;
  return false;
}

bool G2DBlit(const G2DBuffer* src, G2DBuffer* dst, bool letterbox,
             uint32_t letterbox_color, bool prefilled) {
  (void)src;
  (void)dst;
  (void)letterbox;
  (void)letterbox_color;
  (void)prefilled;
  return false;
}

bool G2DWrapDmaBuf(G2DBuffer* buf, int dmabuf_fd, void* virt_addr,
                   int width, int height, uint32_t format) {
  (void)buf;
  (void)dmabuf_fd;
  (void)virt_addr;
  (void)width;
  (void)height;
  (void)format;
  return false;
}

bool G2DResizeToDmaBuf(const uint8_t* src_data, int src_width, int src_height,
                       int src_channels, int dmabuf_fd, void* dmabuf_virt,
                       int dst_width, int dst_height, uint32_t dst_format,
                       bool letterbox, uint32_t letterbox_color, bool prefilled) {
  (void)src_data;
  (void)src_width;
  (void)src_height;
  (void)src_channels;
  (void)dmabuf_fd;
  (void)dmabuf_virt;
  (void)dst_width;
  (void)dst_height;
  (void)dst_format;
  (void)letterbox;
  (void)letterbox_color;
  (void)prefilled;
  return false;
}

bool G2DContextOpen(G2DContext* ctx) { (void)ctx; return false; }
void G2DContextClose(G2DContext* ctx) { (void)ctx; }
bool G2DContextAllocSource(G2DContext* ctx, int w, int h, uint32_t fmt) {
  (void)ctx; (void)w; (void)h; (void)fmt; return false;
}
void G2DContextFreeSource(G2DContext* ctx) { (void)ctx; }
bool G2DContextCachePhysAddr(G2DContext* ctx, int dmabuf_fd) {
  (void)ctx; (void)dmabuf_fd; return false;
}
bool G2DResizeCtx(G2DContext* ctx, const uint8_t* src_data,
                  int src_w, int src_h, int src_ch,
                  G2DBuffer* dst, bool letterbox,
                  uint32_t letterbox_color, bool prefilled) {
  (void)ctx; (void)src_data; (void)src_w; (void)src_h; (void)src_ch;
  (void)dst; (void)letterbox; (void)letterbox_color; (void)prefilled;
  return false;
}
bool G2DResizeToDmaBufCtx(G2DContext* ctx, const uint8_t* src_data,
                           int src_w, int src_h, int src_ch,
                           void* dmabuf_virt, int dst_w, int dst_h,
                           uint32_t dst_fmt, bool letterbox,
                           uint32_t letterbox_color, bool prefilled) {
  (void)ctx; (void)src_data; (void)src_w; (void)src_h; (void)src_ch;
  (void)dmabuf_virt; (void)dst_w; (void)dst_h; (void)dst_fmt;
  (void)letterbox; (void)letterbox_color; (void)prefilled;
  return false;
}

}  // namespace camera_adaptor_test

#endif  // HAVE_G2D
