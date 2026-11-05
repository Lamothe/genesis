#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "animation.h"
#include "gltf.h"
#include "mesh.h"
#include "skin.h"

using namespace vredit;
namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (false)
bool near(float a, float b) { return std::fabs(a-b) < 0.0001F; }
bool near3(Vec3 a, Vec3 b) {
  return std::fabs(a.x-b.x) < 0.0001F && std::fabs(a.y-b.y) < 0.0001F &&
         std::fabs(a.z-b.z) < 0.0001F;
}

void append32(std::vector<std::uint8_t>& bytes, std::uint32_t v) {
  for (unsigned i = 0; i < 4; ++i) { bytes.push_back(static_cast<std::uint8_t>(v >> (i*8))); }
}
void appendFloat(std::vector<std::uint8_t>& bytes, float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, 4);
  append32(bytes, bits);
}
void append16(std::vector<std::uint8_t>& bytes, std::uint16_t v) {
  bytes.push_back(static_cast<std::uint8_t>(v & 0xFFU));
  bytes.push_back(static_cast<std::uint8_t>(v >> 8U));
}
std::vector<std::uint8_t> glb(std::string json, std::vector<std::uint8_t> bin) {
  while (json.size() % 4) { json += ' '; }
  while (bin.size() % 4) { bin.push_back(0); }
  std::vector<std::uint8_t> bytes;
  append32(bytes, 0x46546c67); append32(bytes, 2);
  append32(bytes, static_cast<std::uint32_t>(12 + 8 + json.size() + 8 + bin.size()));
  append32(bytes, static_cast<std::uint32_t>(json.size())); append32(bytes, 0x4e4f534a);
  bytes.insert(bytes.end(), json.begin(), json.end());
  append32(bytes, static_cast<std::uint32_t>(bin.size())); append32(bytes, 0x004e4942);
  bytes.insert(bytes.end(), bin.begin(), bin.end());
  return bytes;
}
std::string fixture(std::string const& animation, std::string const& roots = "[0]") {
  return R"({"asset":{"version":"2.0"},"scenes":[{"nodes":)" + roots +
    R"(}],"nodes":[{"children":[1],"mesh":0},{},{"name":"outside"}],)"
    R"("meshes":[{"primitives":[{"attributes":{"POSITION":2}},{"attributes":{"POSITION":2}}]}],)"
    R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":16,"byteStride":8},)"
    R"({"buffer":0,"byteOffset":16,"byteLength":24},{"buffer":0,"byteOffset":40,"byteLength":36}],)"
    R"("accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},)"
    R"({"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"},)"
    R"({"bufferView":2,"componentType":5126,"count":3,"type":"VEC3"}])" + animation + "}";
}
std::vector<std::uint8_t> data() {
  std::vector<std::uint8_t> b;
  for (float f : {0.F, 99.F, 2.F, 99.F, 0.F, 0.F, 0.F, 4.F, 2.F, 0.F,
                  0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F, 0.F}) { appendFloat(b, f); }
  return b;
}
std::string anim(std::string mode = "LINEAR", int node = 1, int output = 1) {
  return R"(,"animations":[{"name":"move","samplers":[{"input":0,"output":)" +
    std::to_string(output) + R"(,"interpolation":")" + mode +
    R"("}],"channels":[{"sampler":0,"target":{"node":)" + std::to_string(node) +
    R"(,"path":"translation"}}]}])";
}
// A synthetic two-joint skinned arm. Rest layout:
//   joint 0 at the origin, joint 1 at (0, 1, 0), mesh node at the scene root
//   (a sibling of the armature) with a tall two-triangle strip.
// Bottom vertices bind to joint 0, top vertices to joint 1. The clip rotates
// joint 1 by 90 degrees about Z; with IBM = inverse(rest global) the rest pose
// must be exactly identity, which pins the inverse-bind convention.
std::string skinnedJson(bool twoPrimitives = false) {
  std::string const primitive =
      R"({"attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3},"indices":4})";
  std::string primitives = primitive;
  if (twoPrimitives) {
    primitives += "," + primitive;
  }
  return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,4]}],)"
    R"("nodes":[{"name":"Armature","children":[1]},{"name":"Root","children":[2]},)"
    R"({"name":"Joint0","children":[3]},{"name":"Joint1","translation":[0,1,0]},)"
    R"({"name":"Mesh","mesh":0,"skin":0}],)"
    R"("skins":[{"joints":[2,3],"inverseBindMatrices":5}],)"
    R"("meshes":[{"primitives":[)" + primitives + R"(]}],)"
    R"("animations":[{"name":"swing","samplers":[{"input":6,"output":7,"interpolation":"LINEAR"}],)"
    R"("channels":[{"sampler":0,"target":{"node":3,"path":"rotation"}}]}],)"
    R"("accessors":[)"
    R"({"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"},)"
    R"({"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},)"
    R"({"bufferView":2,"componentType":5123,"count":4,"type":"VEC4"},)"
    R"({"bufferView":3,"componentType":5126,"count":4,"type":"VEC4"},)"
    R"({"bufferView":4,"componentType":5123,"count":6,"type":"SCALAR"},)"
    R"({"bufferView":5,"componentType":5126,"count":2,"type":"MAT4"},)"
    R"({"bufferView":6,"componentType":5126,"count":2,"type":"SCALAR"},)"
    R"({"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"}],)"
    R"("bufferViews":[)"
    R"({"buffer":0,"byteOffset":0,"byteLength":48},)"
    R"({"buffer":0,"byteOffset":48,"byteLength":48},)"
    R"({"buffer":0,"byteOffset":96,"byteLength":32},)"
    R"({"buffer":0,"byteOffset":128,"byteLength":64},)"
    R"({"buffer":0,"byteOffset":192,"byteLength":12},)"
    R"({"buffer":0,"byteOffset":204,"byteLength":128},)"
    R"({"buffer":0,"byteOffset":332,"byteLength":8},)"
    R"({"buffer":0,"byteOffset":340,"byteLength":32}],)"
    R"("buffers":[{"byteLength":372}]})";
}

std::vector<std::uint8_t> skinnedBin() {
  std::vector<std::uint8_t> b;
  for (float v : {-0.5F, 0.0F, 0.0F,  0.5F, 0.0F, 0.0F,
                  -0.5F, 1.5F, 0.0F,  0.5F, 1.5F, 0.0F}) {
    appendFloat(b, v);
  }
  for (int i = 0; i < 4; ++i) {
    appendFloat(b, 0.0F);
    appendFloat(b, 0.0F);
    appendFloat(b, 1.0F);
  }
  for (std::uint16_t j : {std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0},
                          std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0},
                          std::uint16_t{1}, std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0},
                          std::uint16_t{1}, std::uint16_t{0}, std::uint16_t{0}, std::uint16_t{0}}) {
    append16(b, j);
  }
  // Influence slot 0 carries weight 1 for every vertex; the joint *index* in
  // slot 0 (0 for the bottom pair, 1 for the top pair) chooses the bone.
  for (int i = 0; i < 4; ++i) {
    for (float w : {1.0F, 0.0F, 0.0F, 0.0F}) {
      appendFloat(b, w);
    }
  }
  for (std::uint16_t i : {std::uint16_t{0}, std::uint16_t{1}, std::uint16_t{2},
                          std::uint16_t{2}, std::uint16_t{1}, std::uint16_t{3}}) {
    append16(b, i);
  }
  // joint 0 rest global = identity; joint 1 rest global = translate(0,1,0).
  for (float v : {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F}) {
    appendFloat(b, v);
  }
  for (float v : {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, -1.0F, 0.0F, 1.0F}) {
    appendFloat(b, v);
  }
  for (float v : {0.0F, 1.0F}) {
    appendFloat(b, v);
  }
  float const half = std::sqrt(0.5F);
  for (float v : {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, half, half}) {
    appendFloat(b, v);
  }
  return b;
}

void testLoader() {
  Model m; std::string error;
  CHECK(loadGlb(glb(fixture(anim()), data()), m, error, false));
  CHECK(m.nodes.size() == 4); // wrapper and its two synthetic primitive children
  CHECK(m.clips.size() == 1 && m.clips[0].channels.size() == 1);
  if (!m.clips.empty() && !m.clips[0].channels.empty()) {
    CHECK(m.clips[0].channels[0].node == 3);
    std::vector<Transform> pose;
    CHECK(evaluateAnimation(m, m.clips[0], 1.F, pose));
    CHECK(near(pose[3].translation.x, 2.F) && near(pose[3].translation.y, 1.F));
    CHECK(near(pose[0].translation.x, 0.F));
  }
  CHECK(loadGlb(glb(fixture(anim("STEP", 2)), data()), m, error, false));
  CHECK(m.clips.empty());
  CHECK(loadGlb(glb(fixture(""), data()), m, error, false) && m.clips.empty());
  CHECK(!loadGlb(glb(fixture(anim(), "[0,1]"), data()), m, error, false));
  CHECK(m.nodes.empty());
  CHECK(loadGlb(glb(fixture(anim("BOGUS")), data()), m, error, false) && m.clips.empty());
  CHECK(loadGlb(glb(fixture(anim("CUBICSPLINE")), data()), m, error, false) && m.clips.empty());
  CHECK(loadGlb(glb(fixture(anim("LINEAR", 99)), data()), m, error, false) && m.clips.empty());
  CHECK(loadGlb(glb(fixture(anim("LINEAR", 1, 9)), data()), m, error, false) && m.clips.empty());
  auto invalidJson = fixture(anim());
  auto index = invalidJson.find("\"count\":2,\"type\":\"SCALAR\"");
  invalidJson.replace(index, std::strlen("\"count\":2,\"type\":\"SCALAR\""),
                      "\"count\":999999999999,\"type\":\"SCALAR\"");
  CHECK(loadGlb(glb(invalidJson, data()), m, error, false) && m.clips.empty());
  invalidJson = fixture(anim());
  index = invalidJson.find("\"children\":[1]");
  invalidJson.insert(index, "\"matrix\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],");
  CHECK(loadGlb(glb(invalidJson, data()), m, error, false)); // node 0 is not targeted
  invalidJson = fixture(anim("LINEAR", 0));
  index = invalidJson.find("\"children\":[1]");
  invalidJson.insert(index, "\"matrix\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],");
  CHECK(loadGlb(glb(invalidJson, data()), m, error, false) && m.clips.empty());
  auto bad = data();
  for (std::size_t i = 8; i < 12; ++i) { bad[i] = 0; } // duplicate time
  CHECK(loadGlb(glb(fixture(anim()), bad), m, error, false) && m.clips.empty());
  bad = data();
  bad[8] = 0; bad[9] = 0; bad[10] = 0xc0; bad[11] = 0x7f; // NaN time
  CHECK(loadGlb(glb(fixture(anim()), bad), m, error, false) && m.clips.empty());
  bad = data();
  bad[16] = 0; bad[17] = 0; bad[18] = 0x80; bad[19] = 0x7f; // infinite output
  CHECK(loadGlb(glb(fixture(anim()), bad), m, error, false) && m.clips.empty());
  bad = data(); bad.resize(20); // out-of-range views
  CHECK(!loadGlb(glb(fixture(anim()), bad), m, error, false));
  auto json = fixture(anim());
  auto pos = json.find("\"count\":2,\"type\":\"SCALAR\"");
  json.replace(pos, std::strlen("\"count\":2,\"type\":\"SCALAR\""),
               "\"count\":2,\"type\":\"SCALAR\",\"sparse\":{}");
  CHECK(loadGlb(glb(json, data()), m, error, false) && m.clips.empty());
  json = fixture(anim());
  pos = json.find("\"count\":2,\"type\":\"SCALAR\"");
  json.replace(pos, std::strlen("\"count\":2,\"type\":\"SCALAR\""),
               "\"count\":2,\"type\":\"SCALAR\",\"byteOffset\":\"bad\"");
  CHECK(loadGlb(glb(json, data()), m, error, false) && m.clips.empty());
}
void testDecodedModes() {
  auto make = [](std::string const& path, std::string const& mode,
                 std::vector<float> const& values, int components) {
    std::vector<std::uint8_t> b;
    appendFloat(b, 0); appendFloat(b, 2);
    for (float v : values) { appendFloat(b, v); }
    std::string json = R"({"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"nodes":[{}],)"
      R"("bufferViews":[{"buffer":0,"byteLength":8},{"buffer":0,"byteOffset":8,"byteLength":)" +
      std::to_string(values.size()*4U) + R"(}],"accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},)"
      R"({"bufferView":1,"componentType":5126,"count":)" + std::to_string(values.size()/static_cast<std::size_t>(components)) +
      R"(,"type":"VEC)" + std::to_string(components) + R"("}],"animations":[{"samplers":[{"input":0,"output":1,"interpolation":")" +
      mode + R"("}],"channels":[{"sampler":0,"target":{"node":0,"path":")" + path + R"("}}]}]})";
    return glb(json, b);
  };
  Model m; std::string error; std::vector<Transform> p;
  auto cubic = make("translation", "CUBICSPLINE",
                    {0,0,0, 0,0,0, 2,0,0, 0,0,0, 4,0,0, 0,0,0}, 3);
  CHECK(loadGlb(cubic, m, error, false));
  CHECK(m.clips.size() == 1 && evaluateAnimation(m, m.clips[0], 1, p));
  CHECK(!p.empty() && near(p[0].translation.x, 2.5F));
  auto rot = make("rotation", "LINEAR", {0,0,0,1, 0,0,1,0}, 4);
  CHECK(loadGlb(rot, m, error, false));
  CHECK(m.clips.size() == 1 && evaluateAnimation(m, m.clips[0], 0.5F, p));
  CHECK(!p.empty() && near(p[0].rotation.z, std::sin(3.14159265F/8)));
  auto invalid = make("scale", "LINEAR", {0,0,0,1, 0,0,1,0}, 4);
  CHECK(loadGlb(invalid, m, error, false) && m.clips.empty());
  invalid = make("weights", "LINEAR", {0,0,0, 1,1,1}, 3);
  CHECK(loadGlb(invalid, m, error, false) && m.clips.empty());
  invalid = make("rotation", "LINEAR", {0,0,0,0, 0,0,1,0}, 4);
  CHECK(loadGlb(invalid, m, error, false) && m.clips.empty());
}
bool matrixNear(Mat4 const& a, Mat4 const& b, float eps = 1e-5F) {
  for (std::size_t i = 0; i < 16U; ++i) {
    if (std::fabs(a.m[i] - b.m[i]) > eps) {
      return false;
    }
  }
  return true;
}

std::vector<Mat4> poseWorlds(Model const& model, std::vector<Transform> const& pose) {
  std::vector<Mat4> worlds(model.nodes.size());
  for (std::size_t n = 0; n < model.nodes.size(); ++n) {
    Mat4 local = toMatrix(pose[n]);
    worlds[n] = model.nodes[n].parent >= 0
                    ? mul(worlds[static_cast<std::size_t>(model.nodes[n].parent)], local)
                    : local;
  }
  return worlds;
}

void testSkinnedModel() {
  Model model;
  std::string error;
  bool const loaded = loadGlb(glb(skinnedJson(), skinnedBin()), model, error, false);
  if (!loaded) {
    std::printf("skinned fixture load error: %s\n", error.c_str());
  }
  CHECK(loaded);
  CHECK(model.skins.size() == 1U);
  if (model.skins.empty()) {
    return;
  }
  CHECK(model.skins[0].joints.size() == 2U);
  if (model.skins.size() == 1U && model.skins[0].joints.size() == 2U) {
    CHECK(model.skins[0].joints[0] == 2 && model.skins[0].joints[1] == 3);
    CHECK(model.skins[0].inverseBindMatrices.size() == 2U);
  }
  CHECK(model.nodes.size() == 5U);
  CHECK(model.nodes[4].skin == 0);
  CHECK(model.nodes[4].skinNode == 4);
  CHECK(model.meshes.size() == 1U);
  CHECK(model.meshes[0].skinWeights.size() == 4U);
  if (model.meshes[0].skinWeights.size() == 4U) {
    CHECK(model.meshes[0].skinWeights[0].joints[0] == 0);
    CHECK(near(model.meshes[0].skinWeights[0].weights[0], 1.0F));
    CHECK(model.meshes[0].skinWeights[2].joints[0] == 1);
    CHECK(near(model.meshes[0].skinWeights[2].weights[0], 1.0F));
  }
  CHECK(model.clips.size() == 1U && model.clips[0].channels.size() == 1U);
  if (model.clips.empty()) {
    return;
  }

  std::vector<Transform> pose;
  CHECK(evaluateAnimation(model, model.clips[0], 0.0F, pose));
  auto restWorlds = poseWorlds(model, pose);
  std::vector<Mat4> restJointMatrices;
  CHECK(computeSkinMatrices(model.skins[0], restWorlds[4],
                            {restWorlds[2], restWorlds[3]}, restJointMatrices));
  CHECK(restJointMatrices.size() == 2U);
  for (Mat4 const& matrix : restJointMatrices) {
    CHECK(matrixNear(matrix, Mat4{}));
  }

  CHECK(evaluateAnimation(model, model.clips[0], 1.0F, pose));
  auto worlds = poseWorlds(model, pose);
  std::vector<Mat4> jointMatrices;
  CHECK(computeSkinMatrices(model.skins[0], worlds[4], {worlds[2], worlds[3]},
                            jointMatrices));
  MeshData deformed = model.meshes[0].data;
  Aabb deformedBounds{};
  CHECK(deformSkinnedMesh(model.meshes[0].data, model.meshes[0].skinWeights,
                          jointMatrices, deformed, &deformedBounds));
  // Fused bounds must equal the bounds of the deformed positions.
  Vec3 expectedMin = deformed.vertices.front().pos;
  Vec3 expectedMax = expectedMin;
  for (MeshVertex const& vertex : deformed.vertices) {
    expectedMin.x = std::min(expectedMin.x, vertex.pos.x);
    expectedMin.y = std::min(expectedMin.y, vertex.pos.y);
    expectedMin.z = std::min(expectedMin.z, vertex.pos.z);
    expectedMax.x = std::max(expectedMax.x, vertex.pos.x);
    expectedMax.y = std::max(expectedMax.y, vertex.pos.y);
    expectedMax.z = std::max(expectedMax.z, vertex.pos.z);
  }
  CHECK(near3(deformedBounds.min, expectedMin) && near3(deformedBounds.max, expectedMax));
  CHECK(near(deformedBounds.max.y, 1.5F));
  CHECK(near3(deformed.vertices[0].pos, {-0.5F, 0.0F, 0.0F}));
  CHECK(near3(deformed.vertices[1].pos, {0.5F, 0.0F, 0.0F}));
  CHECK(near3(deformed.vertices[2].pos, {-0.5F, 0.5F, 0.0F}));
  CHECK(near3(deformed.vertices[3].pos, {-0.5F, 1.5F, 0.0F}));

  // The mesh node transform must cancel exactly in world space: a shifted
  // mesh world changes the local skin result but not the drawn world result.
  Mat4 const shifted = translationMatrix({5.0F, 2.0F, 0.0F});
  std::vector<Mat4> shiftedMatrices;
  CHECK(computeSkinMatrices(model.skins[0], shifted, {worlds[2], worlds[3]},
                            shiftedMatrices));
  MeshData shiftedDeformed = model.meshes[0].data;
  CHECK(deformSkinnedMesh(model.meshes[0].data, model.meshes[0].skinWeights,
                          shiftedMatrices, shiftedDeformed));
  for (std::size_t v = 0; v < deformed.vertices.size(); ++v) {
    CHECK(near3(transformPoint(shifted, shiftedDeformed.vertices[v].pos),
                deformed.vertices[v].pos));
  }
  CHECK(sameSkinMatrices(jointMatrices, jointMatrices));
  CHECK(!sameSkinMatrices(jointMatrices, restJointMatrices));
  CHECK(!sameSkinMatrices({}, {}));

  // End-to-end fixture for --smoke-skinned (best effort).
  std::ofstream fixture(".build/test-skinned.glb", std::ios::binary);
  if (fixture) {
    std::vector<std::uint8_t> bytes = glb(skinnedJson(), skinnedBin());
    fixture.write(reinterpret_cast<char const*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
  }
}

void testSkinNormalsAndFallbacks() {
  MeshData rest;
  rest.vertices.resize(1);
  rest.vertices[0].pos = {1.0F, 2.0F, 3.0F};
  rest.vertices[0].normal = normalized(Vec3{1.0F, 1.0F, 0.0F});
  std::vector<SkinVertexWeight> weights(1);
  weights[0].joints = {0, 0, 0, 0};
  weights[0].weights = {1.0F, 0.0F, 0.0F, 0.0F};

  // Inverse-transpose is required for normals under non-uniform joint scale.
  Mat4 const joint = mul(rotationMatrix(quatFromAxisAngle({0.0F, 0.0F, 1.0F}, 0.7F)),
                         scaleMatrix({2.0F, 1.0F, 0.5F}));
  Mat4 inverse;
  CHECK(invertAffine(joint, inverse));
  Mat4 transposed;
  for (std::size_t row = 0; row < 3U; ++row) {
    for (std::size_t column = 0; column < 3U; ++column) {
      transposed.m[column * 4U + row] = inverse.m[row * 4U + column];
    }
  }
  Vec3 const expectedNormal = normalized(transformVector(transposed, rest.vertices[0].normal));
  MeshData deformed = rest;
  CHECK(deformSkinnedMesh(rest, weights, {joint}, deformed));
  CHECK(near3(deformed.vertices[0].normal, expectedNormal));
  CHECK(near(length(deformed.vertices[0].normal), 1.0F));
  CHECK(near3(deformed.vertices[0].pos, transformPoint(joint, rest.vertices[0].pos)));

  // A singular joint uses the forward 3x3; a normal that collapses to zero
  // keeps its rest direction instead of becoming NaN.
  MeshData singular = rest;
  singular.vertices[0].pos = {0.0F, 0.0F, 0.0F};
  singular.vertices[0].normal = {0.0F, 0.0F, 1.0F};
  CHECK(deformSkinnedMesh(singular, weights, {scaleMatrix({1.0F, 1.0F, 0.0F})},
                          deformed));
  CHECK(near3(deformed.vertices[0].normal, {0.0F, 0.0F, 1.0F}));

  // Out-of-range joints and zero weights leave the bind-pose vertex.
  std::vector<SkinVertexWeight> invalid(1);
  invalid[0].joints = {7, 0, 0, 0};
  invalid[0].weights = {1.0F, 0.0F, 0.0F, 0.0F};
  MeshData untouched = rest;
  CHECK(deformSkinnedMesh(rest, invalid, {Mat4{}}, untouched));
  CHECK(near3(untouched.vertices[0].pos, rest.vertices[0].pos));
  CHECK(near3(untouched.vertices[0].normal, rest.vertices[0].normal));
  CHECK(deformSkinnedMesh(rest, weights, {}, untouched)); // no matrices: rest pose
  CHECK(near3(untouched.vertices[0].pos, rest.vertices[0].pos));

  // A non-finite bind-pose vertex is rejected rather than producing NaN bounds
  // or NaN GPU data (the loader also rejects it; this is the second line).
  MeshData broken = rest;
  broken.vertices[0].pos = {0.0F, std::numeric_limits<float>::quiet_NaN(), 0.0F};
  Aabb brokenBounds{};
  CHECK(!deformSkinnedMesh(broken, weights, {joint}, deformed, &brokenBounds));
}

void testMalformedSkins() {
  auto load = [](std::string json, std::vector<std::uint8_t> bin, Model& model) {
    std::string error;
    return loadGlb(glb(json, bin), model, error, false);
  };
  Model model;

  // JOINTS/WEIGHTS with no skins (the real library's static exports) are
  // ignored entirely: no per-vertex skin allocation.
  std::string noSkins = skinnedJson();
  auto skinAt = noSkins.find(R"("skins":[{"joints":[2,3],"inverseBindMatrices":5}],)");
  CHECK(skinAt != std::string::npos);
  noSkins.erase(skinAt, std::strlen(R"("skins":[{"joints":[2,3],"inverseBindMatrices":5}],)"));
  noSkins.replace(noSkins.find(R"(,"skin":0)"), std::strlen(R"(,"skin":0)"), "");
  CHECK(load(noSkins, skinnedBin(), model));
  CHECK(model.meshes[0].skinWeights.empty());
  CHECK(model.nodes[4].skin == -1);

  // IBM count shorter than the joint array: skin ignored, geometry retained.
  std::string shortIbm = skinnedJson();
  auto ibmCount = shortIbm.find(R"("count":2,"type":"MAT4")");
  CHECK(ibmCount != std::string::npos);
  shortIbm.replace(ibmCount, std::strlen(R"("count":2,"type":"MAT4")"),
                   R"("count":1,"type":"MAT4")");
  CHECK(load(shortIbm, skinnedBin(), model));
  CHECK(model.meshes.size() == 1U && model.skins.size() == 1U &&
        model.skins[0].joints.empty());
  CHECK(model.nodes[4].skin == -1);

  // Joint node outside the selected scene.
  std::string badJoint = skinnedJson();
  badJoint.replace(badJoint.find(R"("joints":[2,3])"), std::strlen(R"("joints":[2,3])"),
                   R"("joints":[2,9])");
  CHECK(load(badJoint, skinnedBin(), model) && model.skins[0].joints.empty());

  // Sparse inverse bind matrices are explicitly unsupported (not silently
  // read as zero matrices).
  std::string sparseIbm = skinnedJson();
  auto ibmAccessor = sparseIbm.find(R"({"bufferView":5,"componentType":5126,"count":2,"type":"MAT4"})");
  CHECK(ibmAccessor != std::string::npos);
  sparseIbm.replace(ibmAccessor,
                    std::strlen(R"({"bufferView":5,"componentType":5126,"count":2,"type":"MAT4"})"),
                    R"({"bufferView":5,"componentType":5126,"count":2,"type":"MAT4","sparse":{}})");
  CHECK(load(sparseIbm, skinnedBin(), model) && model.skins[0].joints.empty());

  // Non-affine inverse bind matrix (bottom row wrong).
  auto badIbm = skinnedBin();
  float const badBottom = 2.0F;
  std::memcpy(badIbm.data() + 204U + 64U + 15U * 4U, &badBottom, 4U);
  CHECK(load(skinnedJson(), badIbm, model) && model.skins[0].joints.empty());

  // WEIGHTS_0 count must match POSITION; the primitive stays rigid.
  std::string shortWeights = skinnedJson();
  auto weightsCount = shortWeights.find(
      R"({"bufferView":3,"componentType":5126,"count":4,"type":"VEC4"})");
  CHECK(weightsCount != std::string::npos);
  shortWeights.replace(
      weightsCount,
      std::strlen(R"({"bufferView":3,"componentType":5126,"count":4,"type":"VEC4"})"),
      R"({"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"})");
  CHECK(load(shortWeights, skinnedBin(), model));
  CHECK(model.meshes[0].skinWeights.empty());

  // Sparse JOINTS_0 is not interpreted.
  std::string sparseJoints = skinnedJson();
  auto jointsAccessor = sparseJoints.find(
      R"({"bufferView":2,"componentType":5123,"count":4,"type":"VEC4"})");
  CHECK(jointsAccessor != std::string::npos);
  sparseJoints.replace(
      jointsAccessor,
      std::strlen(R"({"bufferView":2,"componentType":5123,"count":4,"type":"VEC4"})"),
      R"({"bufferView":2,"componentType":5123,"count":4,"type":"VEC4","sparse":{}})");
  CHECK(load(sparseJoints, skinnedBin(), model));
  CHECK(model.meshes[0].skinWeights.empty());

  // A node pointing at a missing skin renders rigid.
  std::string badSkinIndex = skinnedJson();
  badSkinIndex.replace(badSkinIndex.find(R"("skin":0)"), std::strlen(R"("skin":0)"),
                       R"("skin":9)");
  CHECK(load(badSkinIndex, skinnedBin(), model));
  CHECK(model.nodes[4].skin == -1);

  // Weights are sanitised (normalised, negatives dropped) rather than trusted.
  auto heavy = skinnedBin();
  float const two = 2.0F;
  std::memcpy(heavy.data() + 128U, &two, 4U); // v0 weights [2,0,0,0] -> [1,0,0,0]
  CHECK(load(skinnedJson(), heavy, model));
  CHECK(near(model.meshes[0].skinWeights[0].weights[0], 1.0F));

  // A broken JOINTS_0 index degrades this primitive instead of failing the
  // whole model (the geometry stays usable).
  std::string badJointsIndex = skinnedJson();
  auto jointsAttr = badJointsIndex.find(R"("JOINTS_0":2)");
  CHECK(jointsAttr != std::string::npos);
  badJointsIndex.replace(jointsAttr, std::strlen(R"("JOINTS_0":2)"),
                         R"("JOINTS_0":999)");
  CHECK(load(badJointsIndex, skinnedBin(), model));
  CHECK(model.meshes.size() == 1U && model.meshes[0].data.vertices.size() == 4U);
  CHECK(model.meshes[0].skinWeights.empty());

  // A present but non-numeric inverseBindMatrices is malformed; identity is
  // only the default when the property is absent.
  std::string stringIbm = skinnedJson();
  auto ibmValue = stringIbm.find(R"("inverseBindMatrices":5)");
  CHECK(ibmValue != std::string::npos);
  stringIbm.replace(ibmValue, std::strlen(R"("inverseBindMatrices":5)"),
                    R"("inverseBindMatrices":"nope")");
  CHECK(load(stringIbm, skinnedBin(), model));
  CHECK(model.skins.size() == 1U && model.skins[0].joints.empty());
  CHECK(model.nodes[4].skin == -1);

  // Absent inverseBindMatrices is legal and means identity.
  std::string noIbm = skinnedJson();
  auto ibmProp = noIbm.find(R"(,"inverseBindMatrices":5)");
  CHECK(ibmProp != std::string::npos);
  noIbm.erase(ibmProp, std::strlen(R"(,"inverseBindMatrices":5)"));
  CHECK(load(noIbm, skinnedBin(), model));
  CHECK(model.skins.size() == 1U && model.skins[0].inverseBindMatrices.size() == 2U);
  if (model.skins.size() == 1U && model.skins[0].inverseBindMatrices.size() == 2U) {
    CHECK(matrixNear(model.skins[0].inverseBindMatrices[0], Mat4{}));
  }
}

void testSkinWeightRange() {
  Model model;
  std::string error;
  CHECK(loadGlb(glb(skinnedJson(), skinnedBin()), model, error, false));
  if (model.skins.empty()) {
    return;
  }
  CHECK(skinWeightsMatchSkin(model.meshes[0].skinWeights, model.skins[0]));
  std::vector<SkinVertexWeight> weights(1);
  weights[0].joints = {1, 0, 0, 0};
  weights[0].weights = {1.0F, 0.0F, 0.0F, 0.0F};
  CHECK(skinWeightsMatchSkin(weights, model.skins[0]));
  weights[0].joints = {2, 0, 0, 0};
  CHECK(!skinWeightsMatchSkin(weights, model.skins[0]));
  // A zero-weight slot naming an unknown joint is harmless.
  weights[0].joints = {0, 9, 0, 0};
  weights[0].weights = {1.0F, 0.0F, 0.0F, 0.0F};
  CHECK(skinWeightsMatchSkin(weights, model.skins[0]));
}

void testColourUvAccessorTypes() {
  // MAT4 became a supported accessor type for inverse bind matrices; it must
  // not be accepted as COLOR_0 or TEXCOORD_0. The fixture is extended with a
  // MAT4 accessor of the same element count as POSITION and both attributes
  // point at it: the first three floats of the first matrix are (1,0,0) and
  // the first two (1,0), so a regression is visible in the parsed values.
  std::string json = skinnedJson();
  auto attributes = json.find(
      R"("attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3})");
  CHECK(attributes != std::string::npos);
  json.replace(attributes,
               std::strlen(R"("attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3})"),
               R"("attributes":{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3,)"
               R"("COLOR_0":8,"TEXCOORD_0":8})");
  auto lastAccessor =
      json.find(R"({"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"})");
  CHECK(lastAccessor != std::string::npos);
  json.replace(lastAccessor,
               std::strlen(R"({"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"})"),
               R"({"bufferView":7,"componentType":5126,"count":2,"type":"VEC4"},)"
               R"({"bufferView":8,"componentType":5126,"count":4,"type":"MAT4"})");
  auto lastView = json.find(R"({"buffer":0,"byteOffset":340,"byteLength":32}])");
  CHECK(lastView != std::string::npos);
  std::size_t const afterView =
      json.find(']', lastView + std::strlen(R"({"buffer":0,"byteOffset":340,"byteLength":32})"));
  CHECK(afterView != std::string::npos);
  json.insert(afterView,
              R"(,{"buffer":0,"byteOffset":0,"byteLength":256})");
  Model model;
  std::string error;
  CHECK(loadGlb(glb(json, skinnedBin()), model, error, false));
  CHECK(model.meshes.size() == 1U);
  if (!model.meshes.empty() && !model.meshes[0].data.vertices.empty()) {
    CHECK(near3(model.meshes[0].data.vertices[0].colour, {1.0F, 1.0F, 1.0F}));
    CHECK(near(model.meshes[0].data.vertices[0].uv.x, 0.0F));
    CHECK(near(model.meshes[0].data.vertices[0].uv.y, 0.0F));
  }
}

void testSkinnedMultiPrimitive() {
  Model model;
  std::string error;
  CHECK(loadGlb(glb(skinnedJson(true), skinnedBin()), model, error, false));
  CHECK(model.meshes.size() == 2U);
  CHECK(model.nodes.size() == 7U); // armature 4, wrapper, two primitive children
  if (model.nodes.size() == 7U) {
    CHECK(model.nodes[4].mesh == -1 && model.nodes[4].skin == 0);
    CHECK(model.nodes[5].mesh >= 0 && model.nodes[5].skin == 0);
    CHECK(model.nodes[5].skinNode == 4);
    CHECK(model.nodes[6].mesh >= 0 && model.nodes[6].skin == 0);
    CHECK(model.nodes[6].skinNode == 4);
  }
}

void testScenePoseLayer() {
  Scene scene;
  auto const node = scene.create();
  Transform base; base.translation = {2,0,0};
  scene.setLocal(node, base);
  auto const before = scene.snapshot();
  auto const epoch = scene.epoch();
  scene.setPoseOffset(node, translationMatrix({3,0,0}));
  CHECK(near(scene.worldPosition(node).x, 5));
  CHECK(scene.epoch() == epoch && sameSnapshot(scene.snapshot(), before));
  scene.destroy(node);
  scene.restore(before);
  CHECK(near(scene.worldPosition(node).x, 5));
  scene.setPoseOffset(node, Mat4{});
  CHECK(near(scene.worldPosition(node).x, 2));
}

void testEvaluation() {
  Model m; m.nodes.resize(2); m.nodes[0].local.scale = {2, 3, 4};
  AnimationClip clip; AnimationChannel c;
  c.node = 1; c.times = {1, 3}; c.values = {{0,0,0,0},{4,2,0,0}};
  clip.channels = {c}; std::vector<Transform> p;
  CHECK(evaluateAnimation(m, clip, -1, p) && near(p[1].translation.x, 0));
  CHECK(evaluateAnimation(m, clip, 2, p) && near(p[1].translation.x, 2));
  CHECK(near(p[0].scale.y, 3));
  CHECK(evaluateAnimation(m, clip, 99, p) && near(p[1].translation.x, 4));
  clip.channels[0].interpolation = AnimationInterpolation::Step;
  CHECK(evaluateAnimation(m, clip, 2.99F, p) && near(p[1].translation.x, 0));
  CHECK(evaluateAnimation(m, clip, 3, p) && near(p[1].translation.x, 4));
  c.path = AnimationPath::Rotation; c.times = {0, 1};
  c.values = {{0,0,0,1},{0,0,1,0}}; clip.channels = {c};
  CHECK(evaluateAnimation(m, clip, 0.25F, p));
  CHECK(near(p[1].rotation.z, std::sin(3.14159265F/8)));
  CHECK(near(p[1].rotation.w, std::cos(3.14159265F/8)));
  c.values = {{0,0,0,1},{0,0,0,-1}}; clip.channels = {c};
  CHECK(evaluateAnimation(m, clip, 0.5F, p) && near(std::fabs(p[1].rotation.w), 1));
  c.path = AnimationPath::Translation; c.interpolation = AnimationInterpolation::CubicSpline;
  c.times = {0, 2}; c.values = {{},{0,0,0,0},{2,0,0,0},{0,0,0,0},{4,0,0,0},{}};
  clip.channels = {c};
  CHECK(evaluateAnimation(m, clip, 1, p) && near(p[1].translation.x, 2.5F));
  c.path = AnimationPath::Rotation;
  c.values = {{},{0,0,0,1},{},{},{0,0,1,0},{}}; clip.channels = {c};
  CHECK(evaluateAnimation(m, clip, 1, p) && near(p[1].rotation.z, std::sqrt(0.5F)));
  auto old = p;
  clip.channels[0].times = {0, 0};
  CHECK(!evaluateAnimation(m, clip, 1, p) && p.size() == old.size() &&
        near(p[1].rotation.z, old[1].rotation.z));
  CHECK(!evaluateAnimation(m, clip, std::numeric_limits<float>::quiet_NaN(), p));
}
}
int main() {
  testLoader(); testDecodedModes(); testSkinnedModel(); testSkinNormalsAndFallbacks();
  testMalformedSkins(); testSkinWeightRange(); testColourUvAccessorTypes();
  testSkinnedMultiPrimitive(); testScenePoseLayer(); testEvaluation();
  std::printf("animation tests: %s\n", failures ? "FAILED" : "passed");
  return failures ? 1 : 0;
}
