#include "mesh.h"

#include <cmath>

namespace vredit {

Aabb meshBounds(MeshData const& mesh) {
  if (mesh.vertices.empty()) {
    return {};
  }
  Aabb box{mesh.vertices[0].pos, mesh.vertices[0].pos};
  for (MeshVertex const& vertex : mesh.vertices) {
    box.min = {std::fmin(box.min.x, vertex.pos.x), std::fmin(box.min.y, vertex.pos.y),
               std::fmin(box.min.z, vertex.pos.z)};
    box.max = {std::fmax(box.max.x, vertex.pos.x), std::fmax(box.max.y, vertex.pos.y),
               std::fmax(box.max.z, vertex.pos.z)};
  }
  return box;
}

} // namespace vredit
