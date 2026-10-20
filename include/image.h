#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vredit {

// Asset files (models and images) are read whole into memory; this cap keeps a
// corrupt or absurd file from turning into a std::bad_alloc that terminates
// the process (a real crash seen during on-the-fly spawning). Shared by every
// loader so the limit has one source of truth. 2 GiB is far above any real
// asset.
constexpr std::uint64_t kMaxAssetFileSize = 2ULL * 1024 * 1024 * 1024;

// Decoded image in straight-alpha RGBA8 (row-major, top row first).
struct Image {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::vector<std::uint8_t> rgba;
};

// Decodes PNG (all colour types, bit depths 1/2/4/8/16, tRNS, non-interlaced)
// and baseline JPEG (8-bit, Huffman, no progressive). Format is detected from
// the magic bytes. On failure the output is reset and error is set.
bool decodeImage(std::uint8_t const* bytes, std::size_t size, Image& out, std::string& error);

// Builds the derived mip levels below `base`: each halves the dimensions
// (floor-divide per the Vulkan mip extent rule) with a 2x2 box filter, down
// to 1x1. Level 0 (`base` itself) is not copied — callers upload it directly.
std::vector<Image> buildMipLevels(Image const& base);

// Bilinear upscale to power-of-two dimensions (a no-op for POT images).
// Vulkan workaround: NPOT image extents fault the NVIDIA DMA path in some
// allocation states (VK_ERROR_DEVICE_LOST; seen on 12288^2 and 1024x1000
// textures after other buffers were allocated), while POT images are fine in
// every state. Upscaling loses no image detail.
Image upscaleToPot(Image const& base);

} // namespace vredit
