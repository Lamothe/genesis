#include "skin.h"

#include <cmath>

namespace vredit {
namespace {

bool finiteMatrix(Mat4 const& m) {
  for (float const value : m.m) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

// Inverse-transpose of the affine matrix's upper 3x3, in Mat4 column-major
// form (translation stays zero). A singular joint keeps its forward 3x3: the
// shader's singular fallback uses the same conservative rule.
Mat4 normalMatrix(Mat4 const& m) {
  Vec3 const c0{m.m[0], m.m[1], m.m[2]};
  Vec3 const c1{m.m[4], m.m[5], m.m[6]};
  Vec3 const c2{m.m[8], m.m[9], m.m[10]};
  float const det = dot(c0, cross(c1, c2));
  Mat4 out{};
  if (!(std::fabs(det) > 1e-12F)) {
    out.m[0] = m.m[0];
    out.m[1] = m.m[1];
    out.m[2] = m.m[2];
    out.m[4] = m.m[4];
    out.m[5] = m.m[5];
    out.m[6] = m.m[6];
    out.m[8] = m.m[8];
    out.m[9] = m.m[9];
    out.m[10] = m.m[10];
    return out;
  }
  float const inv = 1.0F / det;
  Vec3 const r0 = cross(c1, c2) * inv; // rows of the inverse
  Vec3 const r1 = cross(c2, c0) * inv;
  Vec3 const r2 = cross(c0, c1) * inv;
  // Transposed: the inverse's rows become the columns.
  out.m[0] = r0.x;
  out.m[1] = r0.y;
  out.m[2] = r0.z;
  out.m[4] = r1.x;
  out.m[5] = r1.y;
  out.m[6] = r1.z;
  out.m[8] = r2.x;
  out.m[9] = r2.y;
  out.m[10] = r2.z;
  return out;
}

} // namespace

bool computeSkinMatrices(ModelSkin const& skin,
                         Mat4 const& meshWorld,
                         std::vector<Mat4> const& jointWorlds,
                         std::vector<Mat4>& out) {
  std::size_t const count = skin.joints.size();
  if (count == 0U || skin.inverseBindMatrices.size() != count ||
      jointWorlds.size() != count) {
    return false;
  }
  Mat4 inverseMesh;
  if (!invertAffine(meshWorld, inverseMesh) || !finiteMatrix(meshWorld) ||
      !finiteMatrix(inverseMesh)) {
    return false;
  }
  std::vector<Mat4> matrices;
  matrices.reserve(count);
  for (std::size_t j = 0; j < count; ++j) {
    if (!finiteMatrix(jointWorlds[j]) || !finiteMatrix(skin.inverseBindMatrices[j])) {
      return false;
    }
    Mat4 matrix = mul(inverseMesh, mul(jointWorlds[j], skin.inverseBindMatrices[j]));
    if (!finiteMatrix(matrix)) {
      return false;
    }
    matrices.push_back(matrix);
  }
  out = std::move(matrices);
  return true;
}

bool deformSkinnedMesh(MeshData const& rest,
                       std::vector<SkinVertexWeight> const& weights,
                       std::vector<Mat4> const& jointMatrices,
                       MeshData& out,
                       Aabb* boundsOut) {
  if (rest.vertices.size() != weights.size()) {
    return false;
  }
  if (out.vertices.size() != rest.vertices.size() || out.indices != rest.indices) {
    out = rest;
  }

  std::vector<Mat4> normals;
  normals.reserve(jointMatrices.size());
  for (Mat4 const& matrix : jointMatrices) {
    if (!finiteMatrix(matrix)) {
      return false;
    }
    normals.push_back(normalMatrix(matrix));
  }

  Aabb bounds{};
  if (!rest.vertices.empty()) {
    bounds.min = rest.vertices.front().pos;
    bounds.max = rest.vertices.front().pos;
  }
  auto extend = [&bounds](Vec3 p) {
    bounds.min.x = std::min(bounds.min.x, p.x);
    bounds.min.y = std::min(bounds.min.y, p.y);
    bounds.min.z = std::min(bounds.min.z, p.z);
    bounds.max.x = std::max(bounds.max.x, p.x);
    bounds.max.y = std::max(bounds.max.y, p.y);
    bounds.max.z = std::max(bounds.max.z, p.z);
  };

  for (std::size_t v = 0; v < rest.vertices.size(); ++v) {
    MeshVertex const& source = rest.vertices[v];
    if (!std::isfinite(source.pos.x) || !std::isfinite(source.pos.y) ||
        !std::isfinite(source.pos.z)) {
      return false; // a non-finite bind-pose vertex cannot be skinned
    }
    MeshVertex& target = out.vertices[v];
    target.pos = source.pos;
    target.normal = source.normal;

    Vec3 position{};
    Vec3 normal{};
    double total = 0.0;
    for (std::size_t k = 0; k < 4U; ++k) {
      float const weight = weights[v].weights[k];
      std::size_t const joint = weights[v].joints[k];
      if (!(weight > 0.0F) || !std::isfinite(weight) || joint >= jointMatrices.size()) {
        continue;
      }
      position = position + transformPoint(jointMatrices[joint], source.pos) * weight;
      normal = normal + transformVector(normals[joint], source.normal) * weight;
      total += static_cast<double>(weight);
    }
    if (total <= 1e-8) {
      extend(target.pos); // no usable influence: the bind-pose vertex
      continue;
    }
    float const scale = static_cast<float>(1.0 / total);
    position = position * scale;
    normal = normal * scale;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
      return false;
    }
    float const normalLength = length(normal);
    target.pos = position;
    extend(position);
    if (std::isfinite(normalLength) && normalLength > 1e-12F) {
      target.normal = normal / normalLength;
    }
  }
  if (boundsOut != nullptr) {
    *boundsOut = bounds;
  }
  return true;
}

bool sameSkinMatrices(std::vector<Mat4> const& a,
                      std::vector<Mat4> const& b,
                      float epsilon) {
  if (a.empty() || a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t k = 0; k < 16U; ++k) {
      if (std::fabs(a[i].m[k] - b[i].m[k]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

bool skinWeightsMatchSkin(std::vector<SkinVertexWeight> const& weights,
                          ModelSkin const& skin) {
  std::size_t const jointCount = skin.joints.size();
  for (SkinVertexWeight const& weight : weights) {
    for (std::size_t k = 0; k < 4U; ++k) {
      if (weight.weights[k] > 0.0F && weight.joints[k] >= jointCount) {
        return false;
      }
    }
  }
  return true;
}

} // namespace vredit