#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "animation.h"
#include "gltf.h"
#include "mesh.h"

using namespace vredit;
namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (false)
bool near(float a, float b) { return std::fabs(a-b) < 0.0001F; }

void append32(std::vector<std::uint8_t>& bytes, std::uint32_t v) {
  for (unsigned i = 0; i < 4; ++i) { bytes.push_back(static_cast<std::uint8_t>(v >> (i*8))); }
}
void appendFloat(std::vector<std::uint8_t>& bytes, float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, 4);
  append32(bytes, bits);
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
  testLoader(); testDecodedModes(); testScenePoseLayer(); testEvaluation();
  std::printf("animation tests: %s\n", failures ? "FAILED" : "passed");
  return failures ? 1 : 0;
}
