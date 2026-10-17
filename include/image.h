#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vredit {

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

// Builds the derived mip levels below `base`: each halves the dimensions (odd
// sizes ceil-divide) with a 2x2 box filter, down to 1x1. Level 0 (`base`
// itself) is not copied — callers upload it directly.
std::vector<Image> buildMipLevels(Image const& base);

} // namespace vredit
