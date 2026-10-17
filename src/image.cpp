#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <zlib.h>

namespace vredit {

namespace {

bool fail(std::string& error, char const* message) {
  error = message;
  return false;
}

std::uint32_t readU32BE(std::uint8_t const* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24U) | (static_cast<std::uint32_t>(p[1]) << 16U) |
         (static_cast<std::uint32_t>(p[2]) << 8U) | static_cast<std::uint32_t>(p[3]);
}

std::uint16_t readU16BE(std::uint8_t const* p) {
  return static_cast<std::uint16_t>((static_cast<std::uint32_t>(p[0]) << 8U) |
                                    static_cast<std::uint32_t>(p[1]));
}

// ---------------------------------------------------------------- PNG ----

std::uint8_t paethPredictor(int a, int b, int c) {
  int p = a + b - c;
  int pa = p > a ? p - a : a - p;
  int pb = p > b ? p - b : b - p;
  int pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) {
    return static_cast<std::uint8_t>(a);
  }
  return static_cast<std::uint8_t>(pb <= pc ? b : c);
}

bool decodePng(std::uint8_t const* bytes, std::size_t size, Image& out, std::string& error) {
  static std::uint8_t const kSignature[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
  if (size < 8U || std::memcmp(bytes, kSignature, 8) != 0) {
    return fail(error, "png: bad signature");
  }

  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t bitDepth = 0;
  std::uint8_t colourType = 0;
  std::uint8_t interlace = 0;
  std::vector<std::uint8_t> palette;
  std::vector<std::uint8_t> transparency;
  std::vector<std::uint8_t> compressed;
  bool sawIhdr = false;

  std::size_t pos = 8U;
  while (pos + 8U <= size) {
    std::uint32_t chunkLength = readU32BE(bytes + pos);
    std::uint32_t chunkType = readU32BE(bytes + pos + 4U);
    pos += 8U;
    if (chunkLength > size || pos + chunkLength + 4U > size) {
      return fail(error, "png: truncated chunk");
    }
    std::uint8_t const* chunkData = bytes + pos;

    // Chunk CRC covers the type and data bytes.
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, bytes + pos - 4U, 4U);
    crc = crc32(crc, chunkData, chunkLength);
    if (static_cast<std::uint32_t>(crc) != readU32BE(bytes + pos + chunkLength)) {
      return fail(error, "png: bad chunk CRC");
    }
    pos += chunkLength + 4U;

    if (chunkType == 0x49484452U) { // 'IHDR'
      if (chunkLength != 13U) {
        return fail(error, "png: bad IHDR");
      }
      width = readU32BE(chunkData);
      height = readU32BE(chunkData + 4U);
      bitDepth = chunkData[8];
      colourType = chunkData[9];
      interlace = chunkData[12];
      // Same total-pixel cap as the JPEG path: hostile dimensions must not
      // drive multi-GB allocations.
      if (width == 0U || height == 0U || width > 65536U || height > 65536U ||
          static_cast<std::uint64_t>(width) * height > 67108864ULL) {
        return fail(error, "png: bad dimensions");
      }
      if (chunkData[10] != 0U || chunkData[11] != 0U) {
        return fail(error, "png: bad compression/filter method");
      }
      if (interlace != 0U) {
        return fail(error, "png: interlaced images are not supported");
      }
      sawIhdr = true;
    } else if (chunkType == 0x504C5445U) { // 'PLTE'
      palette.assign(chunkData, chunkData + chunkLength);
    } else if (chunkType == 0x74524E53U) { // 'tRNS'
      transparency.assign(chunkData, chunkData + chunkLength);
    } else if (chunkType == 0x49444154U) { // 'IDAT'
      compressed.insert(compressed.end(), chunkData, chunkData + chunkLength);
    } else if (chunkType == 0x49454E44U) { // 'IEND'
      break;
    }
    // Ancillary chunks are ignored.
  }

  if (!sawIhdr) {
    return fail(error, "png: missing IHDR");
  }
  if (compressed.empty()) {
    return fail(error, "png: missing image data");
  }

  std::uint32_t channels = 0;
  switch (colourType) {
  case 0: channels = 1; break; // greyscale
  case 2: channels = 3; break; // RGB
  case 3: channels = 1; break; // palette index
  case 4: channels = 2; break; // greyscale + alpha
  case 6: channels = 4; break; // RGBA
  default:
    return fail(error, "png: bad colour type");
  }
  bool depthOk = bitDepth == 8U || bitDepth == 16U ||
                 (bitDepth < 8U && (colourType == 0U || colourType == 3U));
  if (!depthOk || bitDepth == 0U || (bitDepth & (bitDepth - 1U)) != 0U) {
    return fail(error, "png: bad bit depth");
  }
  if (colourType == 3U && palette.empty()) {
    return fail(error, "png: palette image without PLTE");
  }

  std::size_t rowBytes = (static_cast<std::size_t>(width) * channels * bitDepth + 7U) / 8U;
  std::size_t bpp = (channels * bitDepth + 7U) / 8U;
  if (bpp == 0U) {
    bpp = 1U;
  }
  std::size_t expected = static_cast<std::size_t>(height) * (rowBytes + 1U);

  std::vector<std::uint8_t> raw(expected);
  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(compressed.data());
  stream.avail_in = static_cast<uInt>(compressed.size());
  stream.next_out = raw.data();
  stream.avail_out = static_cast<uInt>(raw.size());
  if (inflateInit(&stream) != Z_OK) {
    return fail(error, "png: inflate init failed");
  }
  int result = inflate(&stream, Z_FINISH);
  std::size_t produced = stream.total_out;
  inflateEnd(&stream);
  if (result != Z_STREAM_END && !(result == Z_OK && produced == expected)) {
    return fail(error, "png: bad compressed data");
  }
  if (produced != expected) {
    return fail(error, "png: unexpected data size");
  }

  // Unfilter in place, row by row.
  std::vector<std::uint8_t> previousRow(rowBytes, 0);
  std::uint8_t* rowBase = raw.data();
  out.width = width;
  out.height = height;
  out.rgba.resize(static_cast<std::size_t>(width) * height * 4U);

  for (std::uint32_t y = 0; y < height; ++y) {
    std::uint8_t filter = rowBase[0];
    std::uint8_t* row = rowBase + 1U;
    rowBase += rowBytes + 1U;
    for (std::size_t x = 0; x < rowBytes; ++x) {
      int left = x >= bpp ? row[x - bpp] : 0;
      int up = previousRow[x];
      int upLeft = x >= bpp ? previousRow[x - bpp] : 0;
      switch (filter) {
      case 0:
        break;
      case 1:
        row[x] = static_cast<std::uint8_t>(row[x] + left);
        break;
      case 2:
        row[x] = static_cast<std::uint8_t>(row[x] + up);
        break;
      case 3:
        row[x] = static_cast<std::uint8_t>(row[x] + ((left + up) >> 1));
        break;
      case 4:
        row[x] = static_cast<std::uint8_t>(row[x] + paethPredictor(left, up, upLeft));
        break;
      default:
        return fail(error, "png: bad filter type");
      }
    }

    // Expand one row to RGBA8.
    std::uint8_t* rgba = out.rgba.data() + static_cast<std::size_t>(y) * width * 4U;
    for (std::uint32_t x = 0; x < width; ++x) {
      // values[] holds raw samples (native bit depth); the tRNS key compares
      // against these, and output bytes are derived below.
      std::uint32_t values[4] = {0, 0, 0, 255};
      if (bitDepth < 8U) {
        std::size_t bitPos = static_cast<std::size_t>(x) * bitDepth;
        std::uint8_t mask = static_cast<std::uint8_t>((1U << bitDepth) - 1U);
        std::uint32_t value = (row[bitPos >> 3U] >> (8U - bitDepth - (bitPos & 7U))) & mask;
        values[0] = value;
      } else if (bitDepth == 8U) {
        std::uint8_t const* pixel = row + static_cast<std::size_t>(x) * channels;
        for (std::uint32_t c = 0; c < channels && c < 4U; ++c) {
          values[c] = pixel[c];
        }
      } else {
        std::uint8_t const* pixel = row + static_cast<std::size_t>(x) * channels * 2U;
        for (std::uint32_t c = 0; c < channels && c < 4U; ++c) {
          values[c] = readU16BE(pixel + c * 2U);
        }
      }
      auto outputByte = [bitDepth](std::uint32_t sample) {
        if (bitDepth == 16U) {
          return sample >> 8U;
        }
        return sample;
      };

      std::uint32_t r = 0;
      std::uint32_t g = 0;
      std::uint32_t b = 0;
      std::uint32_t a = 255;
      switch (colourType) {
      case 0: {
        std::uint32_t grey = values[0];
        if (bitDepth < 8U) {
          grey = (grey * 255U) / ((1U << bitDepth) - 1U);
        } else {
          grey = outputByte(grey);
        }
        r = g = b = grey;
        if (!transparency.empty() && transparency.size() >= 2U &&
            values[0] == readU16BE(transparency.data())) {
          a = 0;
        }
        break;
      }
      case 2:
        r = outputByte(values[0]);
        g = outputByte(values[1]);
        b = outputByte(values[2]);
        if (transparency.size() >= 6U && values[0] == readU16BE(transparency.data()) &&
            values[1] == readU16BE(transparency.data() + 2U) &&
            values[2] == readU16BE(transparency.data() + 4U)) {
          a = 0;
        }
        break;
      case 3: {
        std::uint32_t index = values[0];
        if (index * 3U + 2U >= palette.size()) {
          return fail(error, "png: palette index out of range");
        }
        r = palette[index * 3U];
        g = palette[index * 3U + 1U];
        b = palette[index * 3U + 2U];
        a = index < transparency.size() ? transparency[index] : 255U;
        break;
      }
      case 4:
        r = g = b = outputByte(values[0]);
        a = outputByte(values[1]);
        break;
      case 6:
        r = outputByte(values[0]);
        g = outputByte(values[1]);
        b = outputByte(values[2]);
        a = outputByte(values[3]);
        break;
      default:
        break;
      }

      std::uint32_t offset = x * 4U;
      rgba[offset] = static_cast<std::uint8_t>(r);
      rgba[offset + 1U] = static_cast<std::uint8_t>(g);
      rgba[offset + 2U] = static_cast<std::uint8_t>(b);
      rgba[offset + 3U] = static_cast<std::uint8_t>(a);
    }
    std::memcpy(previousRow.data(), row, rowBytes);
  }
  return true;
}

// --------------------------------------------------------------- JPEG ----

struct JpegHuffman {
  std::uint16_t counts[17] = {};
  std::uint8_t symbols[256] = {};
  int minCode[17] = {};
  int maxCode[17] = {};
  int valPtr[17] = {};

  void build() {
    int code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len) {
      valPtr[len] = k;
      minCode[len] = code;
      if (counts[len] == 0U) {
        maxCode[len] = -1;
      } else {
        code += counts[len];
        maxCode[len] = code - 1;
      }
      k += counts[len];
      code <<= 1;
    }
  }
};

struct JpegComponent {
  int id{0};
  int hFactor{1};
  int vFactor{1};
  int quantTable{0};
  int dcTable{0};
  int acTable{0};
  int blocksX{0};
  int blocksY{0};
  std::vector<float> plane; // blocksY*8 rows of blocksX*8 samples
};

struct JpegBits {
  std::uint8_t const* data{nullptr};
  std::size_t size{0};
  std::size_t pos{0};
  std::uint32_t buffer{0};
  int count{0};
  bool markerHit{false};

  bool readBit(int& bit) {
    if (count == 0) {
      if (pos >= size) {
        return false;
      }
      std::uint8_t byte = data[pos++];
      if (byte == 0xFFU) {
        if (pos >= size) {
          return false;
        }
        if (data[pos] != 0x00U) {
          markerHit = true;
          --pos; // leave the marker for the caller
          return false;
        }
        ++pos; // stuffed 0x00
      }
      buffer = byte;
      count = 8;
    }
    bit = static_cast<int>((buffer >> 7U) & 1U);
    buffer = (buffer << 1U) & 0xFFU;
    --count;
    return true;
  }

  bool receive(int n, int& value) {
    value = 0;
    for (int i = 0; i < n; ++i) {
      int bit = 0;
      if (!readBit(bit)) {
        return false;
      }
      value = (value << 1) | bit;
    }
    return true;
  }

  static int extend(int value, int size) {
    return value < (1 << (size - 1)) ? value - (1 << size) + 1 : value;
  }
};

struct JpegFrame {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::vector<JpegComponent> components;
  std::uint16_t quant[4][64] = {};
  bool quantSet[4] = {};
  JpegHuffman dc[4];
  JpegHuffman ac[4];
  bool dcSet[4] = {};
  bool acSet[4] = {};
  int restartInterval{0};
  int adobeTransform{-1};
};

// Zigzag index -> natural (row-major) position in an 8x8 block.
constexpr std::uint8_t kZigzag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

void idct8x8(float block[64]) {
  static float cosTable[8][8] = {};
  static bool tableReady = false;
  if (!tableReady) {
    for (int x = 0; x < 8; ++x) {
      for (int u = 0; u < 8; ++u) {
        cosTable[x][u] =
            std::cos((2.0F * static_cast<float>(x) + 1.0F) * static_cast<float>(u) * 0.19634954F);
      }
    }
    tableReady = true;
  }

  float temp[64];
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      float sum = 0.0F;
      for (int u = 0; u < 8; ++u) {
        float c = u == 0 ? 0.70710678F : 1.0F;
        sum += c * block[y * 8 + u] * cosTable[x][u];
      }
      temp[y * 8 + x] = sum;
    }
  }
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      float sum = 0.0F;
      for (int v = 0; v < 8; ++v) {
        float c = v == 0 ? 0.70710678F : 1.0F;
        sum += c * temp[v * 8 + x] * cosTable[y][v];
      }
      block[y * 8 + x] = 0.25F * sum;
    }
  }
}

bool decodeJpegScanBlock(JpegBits& bits, JpegFrame const& frame, JpegComponent const& component,
                         float* coefficients, int& lastDc, std::string& error) {
  JpegHuffman const& dc = frame.dc[component.dcTable];
  JpegHuffman const& ac = frame.ac[component.acTable];
  for (int i = 0; i < 64; ++i) {
    coefficients[i] = 0.0F;
  }

  int size = 0;
  int code = 0;
  for (int len = 1; len <= 16; ++len) {
    int bit = 0;
    if (!bits.readBit(bit)) {
      return fail(error, "jpeg: truncated entropy data");
    }
    code = (code << 1) | bit;
    if (dc.counts[len] > 0U && code >= dc.minCode[len] && code <= dc.maxCode[len]) {
      size = dc.symbols[dc.valPtr[len] + (code - dc.minCode[len])];
      break;
    }
    if (len == 16) {
      return fail(error, "jpeg: bad DC Huffman code");
    }
  }
  // DC symbols are bit counts: unvalidated values would shift UB into
  // receive/extend. Baseline caps them at 11; 16 is the hard safety limit.
  if (size > 16) {
    return fail(error, "jpeg: bad DC magnitude");
  }
  int diff = 0;
  if (size > 0 && (!bits.receive(size, diff))) {
    return fail(error, "jpeg: truncated entropy data");
  }
  if (size > 0) {
    diff = JpegBits::extend(diff, size);
  }
  lastDc += diff;
  coefficients[0] = static_cast<float>(lastDc);

  int k = 1;
  while (k < 64) {
    int symbol = 0;
    int code2 = 0;
    bool found = false;
    for (int len = 1; len <= 16; ++len) {
      int bit = 0;
      if (!bits.readBit(bit)) {
        return fail(error, "jpeg: truncated entropy data");
      }
      code2 = (code2 << 1) | bit;
      if (ac.counts[len] > 0U && code2 >= ac.minCode[len] && code2 <= ac.maxCode[len]) {
        symbol = ac.symbols[ac.valPtr[len] + (code2 - ac.minCode[len])];
        found = true;
        break;
      }
    }
    if (!found) {
      return fail(error, "jpeg: bad AC Huffman code");
    }
    int run = symbol >> 4;
    int magnitude = symbol & 0x0F;
    if (magnitude == 0) {
      if (run != 15) {
        break; // EOB
      }
      k += 16;
      continue;
    }
    k += run;
    if (k >= 64) {
      return fail(error, "jpeg: coefficient index out of range");
    }
    int value = 0;
    if (!bits.receive(magnitude, value)) {
      return fail(error, "jpeg: truncated entropy data");
    }
    coefficients[kZigzag[k]] = static_cast<float>(JpegBits::extend(value, magnitude));
    ++k;
  }
  return true;
}

bool decodeJpeg(std::uint8_t const* bytes, std::size_t size, Image& out, std::string& error) {
  if (size < 4U || bytes[0] != 0xFFU || bytes[1] != 0xD8U) {
    return fail(error, "jpeg: bad signature");
  }

  JpegFrame frame;
  std::size_t pos = 2U;
  bool sawSof = false;
  std::vector<std::vector<float>> planes; // component index -> decoded samples
  std::size_t mcusX = 0;
  std::size_t mcusY = 0;

  while (pos + 1U < size) {
    if (bytes[pos] != 0xFFU) {
      return fail(error, "jpeg: expected marker");
    }
    while (pos < size && bytes[pos] == 0xFFU) {
      ++pos; // fill bytes
    }
    if (pos >= size) {
      break;
    }
    std::uint8_t marker = bytes[pos++];
    if (marker == 0xD9U) { // EOI
      break;
    }
    if (marker == 0xD8U) {
      continue;
    }
    if (marker >= 0xD0U && marker <= 0xD7U) {
      continue; // stray restart marker
    }
    if (pos + 2U > size) {
      return fail(error, "jpeg: truncated marker segment");
    }
    std::uint32_t segmentLength = readU16BE(bytes + pos);
    if (segmentLength < 2U || pos + segmentLength > size) {
      return fail(error, "jpeg: bad marker segment length");
    }
    std::uint8_t const* payload = bytes + pos + 2U;
    std::size_t payloadSize = segmentLength - 2U;
    pos += segmentLength;

    if (marker == 0xC0U) { // SOF0: baseline
      if (payloadSize < 6U) {
        return fail(error, "jpeg: bad SOF0");
      }
      if (payload[0] != 8U) {
        return fail(error, "jpeg: only 8-bit precision is supported");
      }
      frame.height = readU16BE(payload + 1U);
      frame.width = readU16BE(payload + 3U);
      int componentCount = payload[5];
      if (frame.width == 0U || frame.height == 0U || frame.width * frame.height > 67108864U) {
        return fail(error, "jpeg: bad dimensions");
      }
      if (componentCount != 1 && componentCount != 3) {
        return fail(error, "jpeg: only greyscale and YCbCr/RGB are supported");
      }
      if (payloadSize < 6U + static_cast<std::size_t>(componentCount) * 3U) {
        return fail(error, "jpeg: bad SOF0 components");
      }
      frame.components.resize(static_cast<std::size_t>(componentCount));
      for (int c = 0; c < componentCount; ++c) {
        JpegComponent& component = frame.components[static_cast<std::size_t>(c)];
        component.id = payload[6U + static_cast<std::size_t>(c) * 3U];
        component.hFactor = payload[7U + static_cast<std::size_t>(c) * 3U] >> 4;
        component.vFactor = payload[7U + static_cast<std::size_t>(c) * 3U] & 0x0F;
        component.quantTable = payload[8U + static_cast<std::size_t>(c) * 3U];
        if (component.hFactor < 1 || component.hFactor > 4 || component.vFactor < 1 ||
            component.vFactor > 4 || component.quantTable > 3) {
          return fail(error, "jpeg: bad sampling factors or quantiser");
        }
      }
      sawSof = true;
    } else if (marker == 0xC2U) {
      return fail(error, "jpeg: progressive JPEG is not supported");
    } else if (marker == 0xDBU) { // DQT
      std::size_t offset = 0;
      while (offset + 1U <= payloadSize) {
        int pq = payload[offset] >> 4;
        int table = payload[offset] & 0x0F;
        ++offset;
        if (table > 3 || pq > 1) {
          return fail(error, "jpeg: bad quantiser table");
        }
        std::size_t entrySize = pq == 0 ? 1U : 2U;
        if (offset + 64U * entrySize > payloadSize) {
          return fail(error, "jpeg: truncated quantiser table");
        }
        for (int i = 0; i < 64; ++i) {
          std::uint32_t raw = pq == 0 ? payload[offset + static_cast<std::size_t>(i)]
                                      : readU16BE(payload + offset + static_cast<std::size_t>(i) * 2U);
          frame.quant[table][kZigzag[i]] = static_cast<std::uint16_t>(raw);
        }
        frame.quantSet[table] = true;
        offset += 64U * entrySize;
      }
    } else if (marker == 0xC4U) { // DHT
      std::size_t offset = 0;
      while (offset + 17U <= payloadSize) {
        int tc = payload[offset] >> 4;
        int table = payload[offset] & 0x0F;
        ++offset;
        if (tc > 1 || table > 3) {
          return fail(error, "jpeg: bad Huffman table id");
        }
        JpegHuffman& huffman = tc == 0 ? frame.dc[table] : frame.ac[table];
        int total = 0;
        for (int len = 1; len <= 16; ++len) {
          huffman.counts[len] = payload[offset + static_cast<std::size_t>(len - 1)];
          total += huffman.counts[len];
        }
        offset += 16U;
        if (total > 256 || offset + static_cast<std::size_t>(total) > payloadSize) {
          return fail(error, "jpeg: bad Huffman table");
        }
        for (int i = 0; i < total; ++i) {
          huffman.symbols[i] = payload[offset + static_cast<std::size_t>(i)];
        }
        offset += static_cast<std::size_t>(total);
        huffman.build();
        // DC symbols double as bit counts downstream; reject values that could
        // overflow the shift maths (baseline max is 11).
        for (int i = 0; i < total; ++i) {
          if (tc == 0 && huffman.symbols[i] > 16) {
            return fail(error, "jpeg: bad DC magnitude in Huffman table");
          }
        }
        if (tc == 0) {
          frame.dcSet[table] = true;
        } else {
          frame.acSet[table] = true;
        }
      }
    } else if (marker == 0xDDU) { // DRI
      if (payloadSize < 2U) {
        return fail(error, "jpeg: bad DRI");
      }
      frame.restartInterval = readU16BE(payload);
    } else if (marker == 0xEEU) { // APP14 Adobe
      if (payloadSize >= 12U && std::memcmp(payload, "Adobe", 5) == 0) {
        frame.adobeTransform = payload[11];
      }
    } else if (marker == 0xDAU) { // SOS: decode one scan
      if (!sawSof) {
        return fail(error, "jpeg: scan before frame header");
      }
      if (payloadSize < 1U) {
        return fail(error, "jpeg: bad SOS");
      }
      int scanCount = payload[0];
      if (payloadSize < 1U + static_cast<std::size_t>(scanCount) * 2U + 3U) {
        return fail(error, "jpeg: bad SOS");
      }
      std::vector<JpegComponent*> scanComponents;
      for (int i = 0; i < scanCount; ++i) {
        int id = payload[1U + static_cast<std::size_t>(i) * 2U];
        int tables = payload[2U + static_cast<std::size_t>(i) * 2U];
        JpegComponent* found = nullptr;
        for (JpegComponent& component : frame.components) {
          if (component.id == id) {
            found = &component;
          }
        }
        if (found == nullptr || (tables >> 4) > 3 || (tables & 0x0F) > 3) {
          return fail(error, "jpeg: bad scan component");
        }
        found->dcTable = tables >> 4;
        found->acTable = tables & 0x0F;
        if (!frame.dcSet[found->dcTable] || !frame.acSet[found->acTable]) {
          return fail(error, "jpeg: scan references missing Huffman tables");
        }
        scanComponents.push_back(found);
      }
      if (scanComponents.empty() || scanCount > static_cast<int>(frame.components.size())) {
        return fail(error, "jpeg: bad scan component count");
      }

      // Frame grid (first scan computes it; later scans reuse it).
      int hMax = 1;
      int vMax = 1;
      for (JpegComponent const& component : frame.components) {
        hMax = std::max(hMax, component.hFactor);
        vMax = std::max(vMax, component.vFactor);
      }
      mcusX = (frame.width + static_cast<std::size_t>(hMax) * 8U - 1U) /
              (static_cast<std::size_t>(hMax) * 8U);
      mcusY = (frame.height + static_cast<std::size_t>(vMax) * 8U - 1U) /
              (static_cast<std::size_t>(vMax) * 8U);
      if (planes.size() != frame.components.size()) {
        planes.resize(frame.components.size());
        for (std::size_t c = 0; c < frame.components.size(); ++c) {
          JpegComponent& component = frame.components[c];
          component.blocksX = static_cast<int>(mcusX) * component.hFactor;
          component.blocksY = static_cast<int>(mcusY) * component.vFactor;
          component.plane.assign(static_cast<std::size_t>(component.blocksX) * 8U *
                                     static_cast<std::size_t>(component.blocksY) * 8U,
                                 0.0F);
          if (!frame.quantSet[component.quantTable]) {
            return fail(error, "jpeg: missing quantiser table");
          }
        }
      }

      // Entropy decode this scan into the component planes.
      JpegBits bits{bytes, size, pos, 0, 0, false};
      bool interleaved = scanCount > 1;
      std::size_t unitX = interleaved ? mcusX : mcusX * static_cast<std::size_t>(scanComponents[0]->hFactor);
      std::size_t unitY = interleaved ? mcusY : mcusY * static_cast<std::size_t>(scanComponents[0]->vFactor);
      std::vector<int> lastDc(scanComponents.size(), 0);
      int mcusSinceRestart = 0;
      for (std::size_t unit = 0; unit < unitX * unitY; ++unit) {
        for (std::size_t c = 0; c < scanComponents.size(); ++c) {
          JpegComponent& component = *scanComponents[c];
          int blockCount = interleaved ? component.hFactor * component.vFactor : 1;
          std::size_t baseX = interleaved ? (unit % unitX) * static_cast<std::size_t>(component.hFactor)
                                          : (unit % unitX);
          std::size_t baseY = interleaved ? (unit / unitX) * static_cast<std::size_t>(component.vFactor)
                                          : (unit / unitX);
          for (int b = 0; b < blockCount; ++b) {
            int bx = static_cast<int>(baseX) + (interleaved ? b % component.hFactor : 0);
            int by = static_cast<int>(baseY) + (interleaved ? b / component.hFactor : 0);
            float coefficients[64] = {};
            if (!decodeJpegScanBlock(bits, frame, component, coefficients, lastDc[c], error)) {
              return false;
            }
            // Dequantise (DQT entries are stored in zigzag order; the block is
            // natural order), then IDCT. JPEG's DCT works on samples shifted
            // by -128, so shift back at plane store.
            float block[64];
            for (int i = 0; i < 64; ++i) {
              block[i] = coefficients[i] * static_cast<float>(frame.quant[component.quantTable][i]);
            }
            idct8x8(block);
            std::size_t stride = static_cast<std::size_t>(component.blocksX) * 8U;
            for (int yy = 0; yy < 8; ++yy) {
              std::size_t rowStart = (static_cast<std::size_t>(by) * 8U + static_cast<std::size_t>(yy)) * stride +
                                     static_cast<std::size_t>(bx) * 8U;
              for (int xx = 0; xx < 8; ++xx) {
                component.plane[rowStart + static_cast<std::size_t>(xx)] = block[yy * 8 + xx] + 128.0F;
              }
            }
          }
        }
        ++mcusSinceRestart;
        if (frame.restartInterval > 0 && mcusSinceRestart == frame.restartInterval &&
            unit + 1U < unitX * unitY) {
          bits.count = 0; // byte align
          if (bits.markerHit) {
            bits.markerHit = false;
          }
          // Skip the RSTn marker.
          while (bits.pos + 1U < bits.size && bytes[bits.pos] == 0xFFU &&
                 bytes[bits.pos + 1U] == 0x00U) {
            bits.pos += 2U; // stuffed bytes left over
          }
          if (bits.pos + 1U >= bits.size || bytes[bits.pos] != 0xFFU ||
              bytes[bits.pos + 1U] < 0xD0U || bytes[bits.pos + 1U] > 0xD7U) {
            return fail(error, "jpeg: missing restart marker");
          }
          bits.pos += 2U;
          std::fill(lastDc.begin(), lastDc.end(), 0);
          mcusSinceRestart = 0;
        }
      }
      pos = bits.pos;
      if (frame.restartInterval > 0) {
        // A trailing restart marker may remain after the last MCU group.
        while (pos + 1U < size && bytes[pos] == 0xFFU && bytes[pos + 1U] >= 0xD0U &&
               bytes[pos + 1U] <= 0xD7U) {
          pos += 2U;
        }
      }
    } else {
      // APPn/COM and reserved markers are skipped (segment already consumed).
    }
  }

  if (planes.empty() || frame.width == 0U) {
    return fail(error, "jpeg: no image data");
  }

  // Upsample (nearest) to full resolution and convert to RGBA8.
  int hMax = 1;
  int vMax = 1;
  for (JpegComponent const& component : frame.components) {
    hMax = std::max(hMax, component.hFactor);
    vMax = std::max(vMax, component.vFactor);
  }
  out.width = frame.width;
  out.height = frame.height;
  out.rgba.resize(static_cast<std::size_t>(frame.width) * frame.height * 4U);
  bool greyscale = frame.components.size() == 1U;
  bool rgbDirect = frame.components.size() == 3U &&
                   ((frame.adobeTransform == 0) ||
                    (frame.adobeTransform < 0 && frame.components[0].id == 'R'));

  for (std::uint32_t y = 0; y < frame.height; ++y) {
    for (std::uint32_t x = 0; x < frame.width; ++x) {
      float samples[3] = {0.0F, 0.0F, 0.0F};
      for (std::size_t c = 0; c < frame.components.size(); ++c) {
        JpegComponent const& component = frame.components[c];
        std::size_t sx = static_cast<std::size_t>(x) * static_cast<std::size_t>(component.hFactor) /
                         static_cast<std::size_t>(hMax);
        std::size_t sy = static_cast<std::size_t>(y) * static_cast<std::size_t>(component.vFactor) /
                         static_cast<std::size_t>(vMax);
        std::size_t stride = static_cast<std::size_t>(component.blocksX) * 8U;
        samples[c] = component.plane[sy * stride + sx];
      }
      float r = 0.0F;
      float g = 0.0F;
      float b = 0.0F;
      if (greyscale) {
        r = g = b = samples[0];
      } else if (rgbDirect) {
        r = samples[0];
        g = samples[1];
        b = samples[2];
      } else {
        float cb = samples[1] - 128.0F;
        float cr = samples[2] - 128.0F;
        r = samples[0] + 1.402F * cr;
        g = samples[0] - 0.344136F * cb - 0.714136F * cr;
        b = samples[0] + 1.772F * cb;
      }
      auto clamp = [](float value) {
        int rounded = static_cast<int>(std::lround(value));
        return static_cast<std::uint8_t>(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
      };
      std::size_t offset = (static_cast<std::size_t>(y) * frame.width + x) * 4U;
      out.rgba[offset] = clamp(r);
      out.rgba[offset + 1U] = clamp(g);
      out.rgba[offset + 2U] = clamp(b);
      out.rgba[offset + 3U] = 255U;
    }
  }
  return true;
}

} // namespace

bool decodeImage(std::uint8_t const* bytes, std::size_t size, Image& out, std::string& error) {
  out = Image{};
  Image decoded;
  bool ok = false;
  if (size >= 8U && bytes[0] == 0x89U && bytes[1] == 0x50U && bytes[2] == 0x4EU &&
      bytes[3] == 0x47U) {
    ok = decodePng(bytes, size, decoded, error);
  } else if (size >= 2U && bytes[0] == 0xFFU && bytes[1] == 0xD8U) {
    ok = decodeJpeg(bytes, size, decoded, error);
  } else {
    return fail(error, "unsupported image format (want PNG or baseline JPEG)");
  }
  if (!ok) {
    out = Image{}; // defined state on failure
    return false;
  }
  out = std::move(decoded);
  return true;
}

} // namespace vredit
