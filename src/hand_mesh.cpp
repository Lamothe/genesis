#include "hand_mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vredit {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr int kSides = 12;

bool usable(Vec3 p, bool valid) {
  // Bound inputs so arithmetic remains finite even with corrupt tracking data.
  return valid && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
         std::fabs(p.x) < 10000.0F && std::fabs(p.y) < 10000.0F &&
         std::fabs(p.z) < 10000.0F;
}

bool plausibleDistance(Vec3 a, Vec3 b, float low, float high) {
  auto const d = length(b - a);
  return d >= low && d <= high;
}

std::uint32_t vertex(MeshData& mesh, Vec3 position, Vec3 normal, Vec3 colour) {
  auto const index = static_cast<std::uint32_t>(mesh.vertices.size());
  mesh.vertices.push_back({position, normal, colour, {}});
  return index;
}

// Orient each face against the outward provisional vertex normals, independent
// of the input hand's handedness or the parametrisation's handedness.
void triangle(MeshData& mesh, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  auto const& va = mesh.vertices[a];
  auto const& vb = mesh.vertices[b];
  auto const& vc = mesh.vertices[c];
  auto const face = cross(vb.pos - va.pos, vc.pos - va.pos);
  if (dot(face, va.normal + vb.normal + vc.normal) < 0.0F) {
    std::swap(b, c);
  }
  mesh.indices.insert(mesh.indices.end(), {a, b, c});
}

// Derive continuous vertex normals from the actual tapered/bent surface.
void smoothNormals(MeshData& mesh, std::size_t firstVertex, std::size_t firstIndex) {
  std::vector<Vec3> sums(mesh.vertices.size() - firstVertex);
  for (auto i = firstIndex; i < mesh.indices.size(); i += 3) {
    auto const a = mesh.indices[i];
    auto const b = mesh.indices[i + 1];
    auto const c = mesh.indices[i + 2];
    auto const face = cross(mesh.vertices[b].pos - mesh.vertices[a].pos,
                            mesh.vertices[c].pos - mesh.vertices[a].pos);
    sums[a - firstVertex] = sums[a - firstVertex] + face;
    sums[b - firstVertex] = sums[b - firstVertex] + face;
    sums[c - firstVertex] = sums[c - firstVertex] + face;
  }
  for (auto i = firstVertex; i < mesh.vertices.size(); ++i) {
    auto const n = sums[i - firstVertex];
    if (length(n) > 1e-12F) {
      mesh.vertices[i].normal = normalized(n);
    }
  }
}

void palm(MeshData& mesh, Vec3 centre, Vec3 x, Vec3 y, Vec3 z,
          float width, float longRadius, float thickness, Vec3 colour) {
  auto const firstVertex = mesh.vertices.size();
  auto const firstIndex = mesh.indices.size();
  auto const bottom = vertex(mesh, centre - y * longRadius, -y, colour);
  constexpr int kRings = 7;
  std::array<std::array<std::uint32_t, kSides>, kRings> rings{};
  for (int r = 0; r < kRings; ++r) {
    auto const latitude = -kPi * 0.5F + kPi * static_cast<float>(r + 1) /
                                             static_cast<float>(kRings + 1);
    for (int s = 0; s < kSides; ++s) {
      auto const angle = 2.0F * kPi * static_cast<float>(s) / kSides;
      auto const cx = std::cos(latitude) * std::cos(angle);
      auto const cy = std::sin(latitude);
      auto const cz = std::cos(latitude) * std::sin(angle);
      auto const position = centre + x * (width * cx) + y * (longRadius * cy) +
                            z * (thickness * cz);
      auto const normal = normalized(x * (cx / width) + y * (cy / longRadius) +
                                     z * (cz / thickness));
      rings[r][s] = vertex(mesh, position, normal, colour);
    }
  }
  auto const top = vertex(mesh, centre + y * longRadius, y, colour);
  for (int s = 0; s < kSides; ++s) {
    auto const n = (s + 1) % kSides;
    triangle(mesh, bottom, rings[0][s], rings[0][n]);
    for (int r = 0; r + 1 < kRings; ++r) {
      triangle(mesh, rings[r][s], rings[r + 1][s], rings[r][n]);
      triangle(mesh, rings[r][n], rings[r + 1][s], rings[r + 1][n]);
    }
    triangle(mesh, rings[kRings - 1][s], top, rings[kRings - 1][n]);
  }
  smoothNormals(mesh, firstVertex, firstIndex);
}

void finger(MeshData& mesh, std::vector<Vec3> const& points, float radius,
            Vec3 preferredAxis, Vec3 colour) {
  auto const firstVertex = mesh.vertices.size();
  auto const firstIndex = mesh.indices.size();
  // The last tracking point represents the outside of the fingertip. A ring
  // before it closes the tapered surface into that point, not beyond it.
  std::vector<Vec3> centres(points.begin(), points.end() - 1);
  centres.push_back(points[points.size() - 2] * 0.3F + points.back() * 0.7F);
  std::vector<std::array<std::uint32_t, kSides>> rings(centres.size());
  Vec3 lastU{};
  for (std::size_t r = 0; r < centres.size(); ++r) {
    auto const previous = r == 0 ? centres[0] : centres[r - 1];
    auto const next = r + 1 == centres.size() ? points.back() : centres[r + 1];
    auto tangent = normalized(next - previous);
    if (length(tangent) < 0.5F) {
      tangent = normalized(next - centres[r]);
    }
    auto const guide = r == 0 ? preferredAxis : lastU;
    auto u = guide - tangent * dot(guide, tangent);
    if (length(u) < 0.001F) {
      auto const fallback = std::fabs(tangent.x) < 0.8F ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
      u = fallback - tangent * dot(fallback, tangent);
    }
    u = normalized(u);
    lastU = u;
    auto const v = cross(tangent, u);
    auto const fraction = static_cast<float>(r) / static_cast<float>(centres.size() - 1);
    auto const ringRadius = radius * (1.0F - 0.4F * fraction);
    for (int s = 0; s < kSides; ++s) {
      auto const angle = 2.0F * kPi * static_cast<float>(s) / kSides;
      auto const normal = u * std::cos(angle) + v * std::sin(angle);
      rings[r][s] = vertex(mesh, centres[r] + normal * ringRadius, normal, colour);
    }
  }
  auto const baseDirection = normalized(points[1] - points[0]);
  auto const base = vertex(mesh, points[0], -baseDirection, colour);
  auto const endDirection = normalized(points.back() - points[points.size() - 2]);
  auto const tip = vertex(mesh, points.back(), endDirection, colour);
  for (int s = 0; s < kSides; ++s) {
    auto const n = (s + 1) % kSides;
    triangle(mesh, base, rings[0][s], rings[0][n]);
    for (std::size_t r = 0; r + 1 < rings.size(); ++r) {
      triangle(mesh, rings[r][s], rings[r][n], rings[r + 1][s]);
      triangle(mesh, rings[r][n], rings[r + 1][n], rings[r + 1][s]);
    }
    triangle(mesh, rings.back()[s], rings.back()[n], tip);
  }
  smoothNormals(mesh, firstVertex, firstIndex);
}

} // namespace

MeshData makeHandMesh(Vec3 const (&positions)[kHandJointCount],
                      bool const (&valid)[kHandJointCount], Vec3 colour) {
  MeshData mesh;
  if (!std::isfinite(colour.x) || !std::isfinite(colour.y) || !std::isfinite(colour.z)) {
    colour = {0.72F, 0.48F, 0.36F};
  }
  auto const good = [&](int i) { return usable(positions[i], valid[i]); };
  auto width = 0.08F;
  Vec3 palmNormal{0, 0, 1};
  if (good(1) && good(7) && good(12) && good(22) &&
      plausibleDistance(positions[7], positions[22], 0.025F, 0.16F) &&
      plausibleDistance(positions[1], positions[12], 0.025F, 0.18F)) {
    auto const x = normalized(positions[7] - positions[22]);
    auto const forward = positions[12] - positions[1];
    auto const projected = forward - x * dot(forward, x);
    if (length(projected) > 0.015F) {
      auto const y = normalized(projected);
      palmNormal = normalized(cross(x, y));
      width = length(positions[7] - positions[22]);
      auto const centre = (positions[1] + (positions[7] + positions[12] + positions[22]) / 3.0F) * 0.5F;
      auto const longRadius = length(projected) * 0.58F;
      palm(mesh, centre, x, y, palmNormal, width * 0.64F, longRadius,
           std::clamp(width * 0.22F, 0.008F, 0.025F), colour);
    }
  }
  constexpr std::array<int, 5> starts{2, 7, 12, 17, 22};
  constexpr std::array<int, 5> counts{4, 4, 4, 4, 4};
  constexpr std::array<float, 5> factors{1.12F, 0.98F, 1.05F, 0.96F, 0.80F};
  for (std::size_t digit = 0; digit < starts.size(); ++digit) {
    std::vector<Vec3> points;
    for (int j = 0; j < counts[digit]; ++j) {
      auto const index = starts[digit] + j;
      if (!good(index) || (!points.empty() &&
                           !plausibleDistance(points.back(), positions[index], 0.002F, 0.12F))) {
        points.clear();
        break;
      }
      points.push_back(positions[index]);
    }
    if (!points.empty()) {
      finger(mesh, points, std::clamp(width * 0.115F * factors[digit], 0.005F, 0.016F),
             palmNormal, colour);
    }
  }
  return mesh;
}

} // namespace vredit
