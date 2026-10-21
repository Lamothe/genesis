#include "geometry.h"

#include <cctype>

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

// Stroke glyphs on a unit cell (baseline y = 0, cap height y = 1, advance
// 0.72). Segments are x1,y1,x2,y2 quads. A HUD font for short words and
// model names — deliberately simple, deliberately typed.
constexpr int kMaxSegs = 12;
struct LineGlyph {
  char c;
  int count;
  float segs[kMaxSegs][4];
};

constexpr LineGlyph kGlyphs[] = {
    {'A', 3, {{0.0F, 0.0F, 0.31F, 1.0F}, {0.31F, 1.0F, 0.62F, 0.0F}, {0.13F, 0.42F, 0.49F, 0.42F}}},
    {'B', 7,
     {{0.0F, 0.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.5F, 0.9F},
      {0.5F, 0.9F, 0.5F, 0.62F},
      {0.5F, 0.62F, 0.0F, 0.55F},
      {0.0F, 0.55F, 0.55F, 0.45F},
      {0.55F, 0.45F, 0.55F, 0.12F},
      {0.55F, 0.12F, 0.0F, 0.0F}}},
    {'C', 5,
     {{0.62F, 0.85F, 0.3F, 1.0F},
      {0.3F, 1.0F, 0.0F, 0.75F},
      {0.0F, 0.75F, 0.0F, 0.25F},
      {0.0F, 0.25F, 0.3F, 0.0F},
      {0.3F, 0.0F, 0.62F, 0.15F}}},
    {'D', 6,
     {{0.0F, 0.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.35F, 1.0F},
      {0.35F, 1.0F, 0.62F, 0.75F},
      {0.62F, 0.75F, 0.62F, 0.25F},
      {0.62F, 0.25F, 0.35F, 0.0F},
      {0.35F, 0.0F, 0.0F, 0.0F}}},
    {'E', 4, {{0.62F, 1.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.62F, 0.0F},
              {0.0F, 0.52F, 0.45F, 0.52F}}},
    {'F', 3, {{0.62F, 1.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F, 0.0F}, {0.0F, 0.52F, 0.45F, 0.52F}}},
    {'G', 7,
     {{0.62F, 0.85F, 0.3F, 1.0F},
      {0.3F, 1.0F, 0.0F, 0.75F},
      {0.0F, 0.75F, 0.0F, 0.25F},
      {0.0F, 0.25F, 0.3F, 0.0F},
      {0.3F, 0.0F, 0.62F, 0.2F},
      {0.62F, 0.2F, 0.62F, 0.45F},
      {0.62F, 0.45F, 0.35F, 0.45F}}},
    {'H', 3, {{0.0F, 0.0F, 0.0F, 1.0F}, {0.62F, 0.0F, 0.62F, 1.0F}, {0.0F, 0.52F, 0.62F, 0.52F}}},
    {'I', 3, {{0.31F, 0.0F, 0.31F, 1.0F}, {0.12F, 1.0F, 0.5F, 1.0F}, {0.12F, 0.0F, 0.5F, 0.0F}}},
    {'J', 4,
     {{0.5F, 1.0F, 0.5F, 0.2F},
      {0.5F, 0.2F, 0.35F, 0.0F},
      {0.35F, 0.0F, 0.12F, 0.08F},
      {0.12F, 0.08F, 0.0F, 0.3F}}},
    {'K', 3, {{0.0F, 0.0F, 0.0F, 1.0F}, {0.62F, 1.0F, 0.0F, 0.45F}, {0.18F, 0.58F, 0.62F, 0.0F}}},
    {'L', 2, {{0.0F, 1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.62F, 0.0F}}},
    {'M', 4,
     {{0.0F, 0.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.31F, 0.5F},
      {0.31F, 0.5F, 0.62F, 1.0F},
      {0.62F, 1.0F, 0.62F, 0.0F}}},
    {'N', 3, {{0.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.62F, 0.0F}, {0.62F, 0.0F, 0.62F, 1.0F}}},
    {'O', 8,
     {{0.0F, 0.15F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 0.15F},
      {0.62F, 0.15F, 0.62F, 0.85F},
      {0.62F, 0.85F, 0.47F, 1.0F},
      {0.47F, 1.0F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.0F, 0.85F},
      {0.0F, 0.85F, 0.0F, 0.15F}}},
    {'P', 5,
     {{0.0F, 0.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.5F, 1.0F},
      {0.5F, 1.0F, 0.62F, 0.82F},
      {0.62F, 0.82F, 0.5F, 0.58F},
      {0.5F, 0.58F, 0.0F, 0.55F}}},
    {'Q', 9,
     {{0.0F, 0.15F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 0.15F},
      {0.62F, 0.15F, 0.62F, 0.85F},
      {0.62F, 0.85F, 0.47F, 1.0F},
      {0.47F, 1.0F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.0F, 0.85F},
      {0.0F, 0.85F, 0.0F, 0.15F},
      {0.4F, 0.28F, 0.62F, 0.0F}}},
    {'R', 6,
     {{0.0F, 0.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.5F, 1.0F},
      {0.5F, 1.0F, 0.62F, 0.82F},
      {0.62F, 0.82F, 0.5F, 0.58F},
      {0.5F, 0.58F, 0.0F, 0.55F},
      {0.35F, 0.55F, 0.62F, 0.0F}}},
    {'S', 7,
     {{0.6F, 0.9F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.0F, 0.75F},
      {0.0F, 0.75F, 0.15F, 0.55F},
      {0.15F, 0.55F, 0.5F, 0.45F},
      {0.5F, 0.45F, 0.62F, 0.25F},
      {0.62F, 0.25F, 0.45F, 0.0F},
      {0.45F, 0.0F, 0.0F, 0.1F}}},
    {'T', 2, {{0.31F, 0.0F, 0.31F, 1.0F}, {0.0F, 1.0F, 0.62F, 1.0F}}},
    {'U', 5,
     {{0.0F, 1.0F, 0.0F, 0.2F},
      {0.0F, 0.2F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 0.2F},
      {0.62F, 0.2F, 0.62F, 1.0F}}},
    {'V', 2, {{0.0F, 1.0F, 0.31F, 0.0F}, {0.31F, 0.0F, 0.62F, 1.0F}}},
    {'W', 4,
     {{0.0F, 1.0F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.31F, 0.6F},
      {0.31F, 0.6F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 1.0F}}},
    {'X', 2, {{0.0F, 0.0F, 0.62F, 1.0F}, {0.0F, 1.0F, 0.62F, 0.0F}}},
    {'Y', 3, {{0.0F, 1.0F, 0.31F, 0.5F}, {0.62F, 1.0F, 0.31F, 0.5F}, {0.31F, 0.5F, 0.31F, 0.0F}}},
    {'Z', 3, {{0.0F, 1.0F, 0.62F, 1.0F}, {0.62F, 1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.62F, 0.0F}}},
    {'0', 9,
     {{0.0F, 0.15F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 0.15F},
      {0.62F, 0.15F, 0.62F, 0.85F},
      {0.62F, 0.85F, 0.47F, 1.0F},
      {0.47F, 1.0F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.0F, 0.85F},
      {0.0F, 0.85F, 0.0F, 0.15F},
      {0.05F, 0.12F, 0.57F, 0.88F}}},
    {'1', 3, {{0.15F, 0.8F, 0.35F, 1.0F}, {0.35F, 1.0F, 0.35F, 0.0F}, {0.15F, 0.0F, 0.55F, 0.0F}}},
    {'2', 5,
     {{0.0F, 0.8F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.45F, 1.0F},
      {0.45F, 1.0F, 0.62F, 0.8F},
      {0.62F, 0.8F, 0.0F, 0.0F},
      {0.0F, 0.0F, 0.62F, 0.0F}}},
    {'3', 6,
     {{0.0F, 1.0F, 0.62F, 1.0F},
      {0.62F, 1.0F, 0.25F, 0.55F},
      {0.25F, 0.55F, 0.5F, 0.5F},
      {0.5F, 0.5F, 0.62F, 0.28F},
      {0.62F, 0.28F, 0.45F, 0.0F},
      {0.45F, 0.0F, 0.0F, 0.0F}}},
    {'4', 3, {{0.45F, 0.0F, 0.45F, 1.0F}, {0.45F, 1.0F, 0.0F, 0.32F}, {0.0F, 0.32F, 0.62F, 0.32F}}},
    {'5', 6,
     {{0.62F, 1.0F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.0F, 0.55F},
      {0.0F, 0.55F, 0.45F, 0.55F},
      {0.45F, 0.55F, 0.62F, 0.3F},
      {0.62F, 0.3F, 0.45F, 0.0F},
      {0.45F, 0.0F, 0.0F, 0.05F}}},
    {'6', 8,
     {{0.55F, 1.0F, 0.15F, 1.0F},
      {0.15F, 1.0F, 0.0F, 0.5F},
      {0.0F, 0.5F, 0.0F, 0.15F},
      {0.0F, 0.15F, 0.15F, 0.0F},
      {0.15F, 0.0F, 0.47F, 0.0F},
      {0.47F, 0.0F, 0.62F, 0.18F},
      {0.62F, 0.18F, 0.5F, 0.42F},
      {0.5F, 0.42F, 0.12F, 0.42F}}},
    {'7', 2, {{0.0F, 1.0F, 0.62F, 1.0F}, {0.62F, 1.0F, 0.25F, 0.0F}}},
    {'8', 7,
     {{0.0F, 0.52F, 0.0F, 1.0F},
      {0.0F, 1.0F, 0.62F, 1.0F},
      {0.62F, 1.0F, 0.62F, 0.52F},
      {0.62F, 0.52F, 0.0F, 0.52F},
      {0.0F, 0.52F, 0.0F, 0.0F},
      {0.0F, 0.0F, 0.62F, 0.0F},
      {0.62F, 0.0F, 0.62F, 0.52F}}},
    {'9', 8,
     {{0.07F, 0.0F, 0.45F, 0.0F},
      {0.45F, 0.0F, 0.62F, 0.5F},
      {0.62F, 0.5F, 0.62F, 0.85F},
      {0.62F, 0.85F, 0.45F, 1.0F},
      {0.45F, 1.0F, 0.13F, 1.0F},
      {0.13F, 1.0F, 0.0F, 0.82F},
      {0.0F, 0.82F, 0.12F, 0.58F},
      {0.12F, 0.58F, 0.5F, 0.58F}}},
    {'-', 1, {{0.08F, 0.5F, 0.54F, 0.5F}}},
    {'.', 1, {{0.27F, 0.0F, 0.35F, 0.0F}}},
    {'/', 1, {{0.0F, 0.0F, 0.62F, 1.0F}}},
    {'_', 1, {{0.05F, -0.12F, 0.57F, -0.12F}}},
};

LineGlyph const* findGlyph(char c) {
  char const u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  for (auto const& g : kGlyphs) {
    if (g.c == u) {
      return &g;
    }
  }
  return nullptr;
}

} // namespace

void appendTextLines(std::vector<RendererVertex>& out, std::string_view text, Mat4 const& place,
                     float height, Vec3 colour) {
  Vec3 const up{0.0F, 1.0F, 0.0F};
  float pen = 0.0F;
  for (char const c : text) {
    LineGlyph const* glyph = findGlyph(c);
    if (glyph == nullptr) {
      pen += 0.72F * height; // spaces and unknown characters
      continue;
    }
    for (int s = 0; s < glyph->count; ++s) {
      Vec3 const a = transformPoint(place, {pen + glyph->segs[s][0] * height,
                                            glyph->segs[s][1] * height, 0.0F});
      Vec3 const b = transformPoint(place, {pen + glyph->segs[s][2] * height,
                                            glyph->segs[s][3] * height, 0.0F});
      out.push_back({a, up, colour, {0.0F, 0.0F}});
      out.push_back({b, up, colour, {0.0F, 0.0F}});
    }
    pen += 0.72F * height;
  }
}

} // namespace vredit
