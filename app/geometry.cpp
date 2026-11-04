#include "geometry.h"

#include "sans_glyphs.h"

#include <cstdint>

namespace vredit {

namespace {

void appendTriangle(std::vector<RendererVertex>& out,
                    Vec3 a,
                    Vec3 b,
                    Vec3 c,
                    Vec3 normal) {
  out.push_back({a, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
  out.push_back({b, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
  out.push_back({c, normal, {1.0F, 1.0F, 1.0F}, {0.0F, 0.0F}});
}

void appendQuad(std::vector<RendererVertex>& out,
                Vec3 a,
                Vec3 b,
                Vec3 c,
                Vec3 d,
                Vec3 normal) {
  appendTriangle(out, a, b, c, normal);
  appendTriangle(out, a, c, d, normal);
}

} // namespace

void appendCube(std::vector<RendererVertex>& out) {
  constexpr float h = 0.5F;
  // Faces wound consistently; culling is disabled in the pipeline so winding
  // only affects normals, which are supplied per face.
  appendQuad(out, {h, -h, -h}, {h, h, -h}, {h, h, h}, {h, -h, h}, {1.0F, 0.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {-h, h, h}, {-h, h, -h}, {-h, -h, -h}, {-1.0F, 0.0F, 0.0F});
  appendQuad(out, {-h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}, {0.0F, 1.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {0.0F, -1.0F, 0.0F});
  appendQuad(out, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {0.0F, 0.0F, 1.0F});
  appendQuad(out, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {0.0F, 0.0F, -1.0F});
}

void appendGrid(std::vector<RendererVertex>& out, float extent, float step) {
  Vec3 const gridColour{0.28F, 0.30F, 0.34F};
  Vec3 const xColour{0.75F, 0.22F, 0.22F};
  Vec3 const zColour{0.22F, 0.38F, 0.80F};
  Vec3 const up{0.0F, 1.0F, 0.0F};
  for (float v = -extent; v <= extent + 0.5F * step; v += step) {
    bool xAxis = (v == 0.0F);
    bool zAxis = (v == 0.0F);
    out.push_back({{-extent, 0.0F, v}, up, xAxis ? xColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{extent, 0.0F, v}, up, xAxis ? xColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{v, 0.0F, -extent}, up, zAxis ? zColour : gridColour, {0.0F, 0.0F}});
    out.push_back({{v, 0.0F, extent}, up, zAxis ? zColour : gridColour, {0.0F, 0.0F}});
  }
}

void appendRayBeam(std::vector<RendererVertex>& out, Ray const& ray, float length, Vec3 colour) {
  Vec3 const up{0.0F, 1.0F, 0.0F};
  Vec3 const dir = normalized(ray.direction);
  Vec3 const tip = ray.origin + dir * length;
  out.push_back({ray.origin, up, colour, {0.0F, 0.0F}});
  out.push_back({tip, up, colour, {0.0F, 0.0F}});
  // Tip cross: two short segments perpendicular to the beam, so the aimed
  // point stays readable when it lands on an untextured surface.
  float const arm = length * 0.03F;
  Vec3 side = cross(dir, up);
  if (dot(side, side) < 1e-8F) {
    side = cross(dir, Vec3{1.0F, 0.0F, 0.0F});
  }
  side = normalized(side);
  Vec3 const lift = normalized(cross(side, dir));
  out.push_back({tip - side * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip + side * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip - lift * arm, up, colour, {0.0F, 0.0F}});
  out.push_back({tip + lift * arm, up, colour, {0.0F, 0.0F}});
}

namespace {
// Decode UTF-8 once per character; unsupported glyphs use a visible '?'.
unsigned nextGlyph(std::string_view text, std::size_t& offset) {
  unsigned const first = static_cast<unsigned char>(text[offset++]);
  if (first < 128) { return first; }
  unsigned count = first >= 0xF0 && first <= 0xF4 ? 3 :
                   first >= 0xE0 && first <= 0xEF ? 2 :
                   first >= 0xC2 && first <= 0xDF ? 1 : 0;
  while (count > 0 && offset < text.size() &&
         (static_cast<unsigned char>(text[offset]) & 0xC0) == 0x80) {
    ++offset;
    --count;
  }
  return '?';
}
} // namespace

void appendTextLines(std::vector<RendererVertex>& out, std::string_view text, Mat4 const& place,
                     float height, Vec3 colour) {
  float pen = 0.0F;
  for (std::size_t offset = 0; offset < text.size();) {
    unsigned c = nextGlyph(text, offset);
    if (c < 32 || c > 126) { c = '?'; }
    for (int y = 0; y < 32; ++y) {
      auto const row = kSansGlyphs[c - 32][y];
      for (int x = 0; x < 24;) {
        if ((row & (1u << x)) == 0) { ++x; continue; }
        int const begin = x;
        while (x < 24 && (row & (1u << x)) != 0) { ++x; }
        float const py = (27.0F - static_cast<float>(y)) * height / 24.0F;
        Vec3 const a = transformPoint(place, {pen + begin * height / 24.0F, py, 0});
        Vec3 const b = transformPoint(place, {pen + x * height / 24.0F, py, 0});
        out.push_back({a, {0,1,0}, colour, {0,0}});
        out.push_back({b, {0,1,0}, colour, {0,0}});
      }
    }
    pen += height;
  }
}

float textLineWidth(std::string_view text, float height) {
  std::size_t count = 0;
  for (std::size_t offset = 0; offset < text.size(); ++count) { (void)nextGlyph(text, offset); }
  return static_cast<float>(count) * height;
}

} // namespace vredit
