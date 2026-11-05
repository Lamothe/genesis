#include "hand_mesh.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace vredit;

namespace {

void makePose(Vec3 (&p)[kHandJointCount], bool (&valid)[kHandJointCount], float mirror) {
  for (auto& v : valid) { v = true; }
  p[0] = {0, 0.04F, 0};
  p[1] = {0, 0, 0};
  p[2] = {mirror * 0.025F, 0.025F, 0};
  p[3] = {mirror * 0.042F, 0.040F, 0};
  p[4] = {mirror * 0.058F, 0.055F, 0};
  p[5] = {mirror * 0.070F, 0.065F, 0};
  for (int finger = 0; finger < 4; ++finger) {
    auto const start = 6 + finger * 5;
    auto const x = mirror * (0.031F - static_cast<float>(finger) * 0.020F);
    p[start] = {x * 0.7F, 0.030F, 0};
    p[start + 1] = {x, 0.075F - finger * 0.002F, 0};
    p[start + 2] = {x, 0.104F - finger * 0.003F, 0.002F};
    p[start + 3] = {x, 0.127F - finger * 0.004F, 0.004F};
    p[start + 4] = {x, 0.146F - finger * 0.005F, 0.006F};
  }
}

void checkMesh(MeshData const& mesh) {
  assert(!mesh.vertices.empty() && !mesh.indices.empty());
  assert(mesh.indices.size() % 3 == 0);
  for (auto const& v : mesh.vertices) {
    assert(std::isfinite(v.pos.x) && std::isfinite(v.pos.y) && std::isfinite(v.pos.z));
    assert(std::isfinite(v.normal.x) && std::isfinite(v.normal.y) && std::isfinite(v.normal.z));
    assert(std::fabs(length(v.normal) - 1.0F) < 0.001F);
  }
  for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
    auto const a = mesh.indices[i];
    auto const b = mesh.indices[i + 1];
    auto const c = mesh.indices[i + 2];
    assert(a < mesh.vertices.size() && b < mesh.vertices.size() && c < mesh.vertices.size());
    auto const& va = mesh.vertices[a];
    auto const& vb = mesh.vertices[b];
    auto const& vc = mesh.vertices[c];
    auto const face = cross(vb.pos - va.pos, vc.pos - va.pos);
    assert(length(face) > 1e-10F);
    assert(dot(face, va.normal + vb.normal + vc.normal) > 0);
  }
}

} // namespace

int main() {
  Vec3 positions[kHandJointCount]{};
  bool valid[kHandJointCount]{};
  assert(makeHandMesh(positions, valid).vertices.empty());
  for (float mirror : {-1.0F, 1.0F}) {
    makePose(positions, valid, mirror);
    auto const open = makeHandMesh(positions, valid);
    checkMesh(open);
    for (auto& p : positions) { p = p + Vec3{0.5F, -0.2F, 1.0F}; }
    auto const shifted = makeHandMesh(positions, valid);
    assert(shifted.vertices.size() == open.vertices.size());
    assert(shifted.indices == open.indices);
    for (std::size_t i = 0; i < open.vertices.size(); ++i) {
      assert(length(shifted.vertices[i].pos - open.vertices[i].pos - Vec3{0.5F, -0.2F, 1.0F}) < 1e-5F);
    }
    makePose(positions, valid, mirror);
    auto const bounds = [&]() {
      auto lo = open.vertices[0].pos;
      auto hi = lo;
      for (auto const& v : open.vertices) {
        lo.x = std::fmin(lo.x, v.pos.x); lo.y = std::fmin(lo.y, v.pos.y); lo.z = std::fmin(lo.z, v.pos.z);
        hi.x = std::fmax(hi.x, v.pos.x); hi.y = std::fmax(hi.y, v.pos.y); hi.z = std::fmax(hi.z, v.pos.z);
      }
      return hi - lo;
    }();
    assert(bounds.x > 0.08F && bounds.x < 0.25F);
    assert(bounds.y > 0.13F && bounds.y < 0.25F);
    assert(bounds.z > 0.02F && bounds.z < 0.1F);
    positions[9].z = 0.02F;
    positions[10].z = 0.04F;
    auto const bent = makeHandMesh(positions, valid);
    checkMesh(bent);
    assert(bent.vertices.size() == open.vertices.size());
    bool changed = false;
    for (std::size_t i = 0; i < open.vertices.size(); ++i) {
      if (length(open.vertices[i].pos - bent.vertices[i].pos) > 0.001F) { changed = true; }
    }
    assert(changed);
    valid[10] = false;
    auto const missing = makeHandMesh(positions, valid);
    checkMesh(missing);
    assert(missing.vertices.size() < open.vertices.size());
    valid[10] = true;
    positions[10].x = std::numeric_limits<float>::quiet_NaN();
    assert(makeHandMesh(positions, valid).vertices.size() == missing.vertices.size());
    for (auto& v : valid) { v = false; }
    assert(makeHandMesh(positions, valid).vertices.empty());
    makePose(positions, valid, mirror);
    positions[10] = positions[9];
    assert(makeHandMesh(positions, valid).vertices.size() < open.vertices.size());
  }
  std::puts("hand mesh checks passed");
}
