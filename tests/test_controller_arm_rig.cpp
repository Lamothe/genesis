#include "controller_arm_rig.h"
#include "gltf.h"
#include "skin.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace vredit;
namespace {

std::int32_t add(Model& m, std::string name, std::int32_t parent, Vec3 translation) {
  ModelNode n;
  n.name = std::move(name); n.parent = parent; n.local.translation = translation;
  m.nodes.push_back(n);
  return static_cast<std::int32_t>(m.nodes.size()-1);
}
Model fixture() {
  Model m;
  auto const root = add(m, "root", -1, {});
  for (int side = 0; side < 2; ++side) {
    float const sign = side == 0 ? -1.0F : 1.0F;
    std::string const suffix = side == 0 ? "_L_1" : "_R_1";
    ModelSkin skin;
    auto const sh = add(m, "Shoulder" + suffix, root, {sign*0.15F, 0, 0});
    auto const part = add(m, "ShoulderPart1" + suffix, sh, {sign*0.1F, 0, 0});
    auto const el = add(m, "Elbow" + suffix, part, {sign*0.1F, -0.04F, 0});
    auto const wr = add(m, "Wrist" + suffix, el, {sign*0.24F, 0, 0});
    skin.joints = {sh, part, el, wr};
    for (auto const* finger : {"IndexFinger", "MiddleFinger", "RingFinger", "PinkyFinger", "ThumbFinger"}) {
      auto parent = wr;
      for (int j = 1; j <= 4; ++j) {
        parent = add(m, std::string(finger) + std::to_string(j) + suffix, parent,
                     {sign * 0.025F, j == 1 ? 0.01F : 0.0F, 0});
        skin.joints.push_back(parent);
      }
    }
    skin.inverseBindMatrices.resize(skin.joints.size());
    m.skins.push_back(std::move(skin));
  }
  return m;
}
std::size_t joint(Model const& m, ControllerArmPose const& p, std::string const& prefix) {
  auto const& js = m.skins[p.skin].joints;
  for (std::size_t i = 0; i < js.size(); ++i) {
    if (m.nodes[static_cast<std::size_t>(js[i])].name.starts_with(prefix)) { return i; }
  }
  assert(false); return 0;
}
Vec3 position(Mat4 const& m) { return {m.m[12],m.m[13],m.m[14]}; }
bool near(Vec3 a, Vec3 b, float e = 1e-4F) { return length(a-b) < e; }
bool same(Mat4 const& a, Mat4 const& b, float e = 1e-5F) {
  for (int i = 0; i < 16; ++i) { if (std::fabs(a.m[i]-b.m[i]) > e) { return false; } }
  return true;
}
// Sweep through the bind upper-arm direction, where projecting that arm onto
// the target plane used to reverse the elbow. Repeat after a rigid placement.
void trajectory(Model const& m, ControllerArmSide side, Transform const& placement, bool depth) {
  float const sign = side == ControllerArmSide::Left ? -1.0F : 1.0F;
  Vec3 const bindShoulder{sign*0.15F, 0, 0};
  Vec3 const bindUpper{sign*0.2F, -0.04F, 0};
  float const upper = length(bindUpper), lower = 0.24F;
  ControllerArmInput input; input.side = side; input.placement = placement;
  Vec3 const shoulder = placement.translation + rotate(placement.rotation, bindShoulder);
  Vec3 previous{};
  for (int i = -40; i <= 40; ++i) {
    Vec3 const localDirection = normalized(bindUpper + (depth ? Vec3{0, 0, i*0.001F} : Vec3{0, i*0.001F, 0}));
    input.grip.translation = shoulder + rotate(placement.rotation, localDirection)*0.32F;
    input.grip.rotation = placement.rotation;
    ControllerArmPose pose;
    assert(solveControllerArm(m, input, pose));
    assert(!pose.wristClamped);
    Vec3 const elbow = position(pose.jointWorlds[joint(m, pose, "Elbow")]);
    Vec3 const wrist = position(pose.jointWorlds[joint(m, pose, "Wrist")]);
    assert(near(wrist, input.grip.translation, 3e-4F));
    assert(std::fabs(length(elbow-shoulder)-upper) < 3e-4F);
    assert(std::fabs(length(wrist-elbow)-lower) < 3e-4F);
    if (i != -40) { assert(length(elbow-previous) < 0.003F); }
    previous = elbow;
  }
}

// Stateful paths through the bind antipode and a loop enclosing it. The
// directions are in placement coordinates; the final pass rotates placement
// every sample to verify that the state follows the body, not world axes.
void antipodeTrajectory(Model const& m, ControllerArmSide side, int path, bool movingPlacement) {
  constexpr float pi = 3.14159265358979323846F;
  float const sign = side == ControllerArmSide::Left ? -1.0F : 1.0F;
  Vec3 const bindDirection = normalized(Vec3{sign*0.44F, -0.04F, 0});
  Vec3 const transverse = normalized(cross(bindDirection, Vec3{0, 0, 1}));
  Vec3 const depth{0, 0, 1};
  float const upper = length(Vec3{sign*0.2F, -0.04F, 0});
  ControllerArmInput input; input.side = side;
  ControllerArmState state;
  Vec3 previousLocal{};
  for (int i = 0; i <= 400; ++i) {
    float const t = static_cast<float>(i) / 400.0F;
    Vec3 direction;
    if (path == 3) {
      float const a = 2.0F*pi*t;
      direction = bindDirection*std::cos(a) + transverse*std::sin(a);
    } else if (path == 2) {
      float const a = 2.0F*pi*t;
      direction = normalized(-bindDirection + (transverse*std::cos(a) + depth*std::sin(a))*0.08F);
    } else {
      float const a = (t - 0.5F)*0.4F * (path == 0 ? 1.0F : -1.0F);
      direction = normalized(-bindDirection*std::cos(a) + (path == 0 ? transverse : depth)*std::sin(a));
    }
    if (movingPlacement) {
      input.placement.rotation = quatFromAxisAngle(normalized(Vec3{1,2,3}), t*1.1F);
      input.placement.translation = {0.4F*t, -0.2F*t, 0.3F*t};
    }
    Vec3 const shoulderLocal{sign*0.15F, 0, 0};
    Vec3 const shoulder = input.placement.translation + rotate(input.placement.rotation, shoulderLocal);
    input.grip.translation = shoulder + rotate(input.placement.rotation, direction)*0.32F;
    input.grip.rotation = input.placement.rotation;
    ControllerArmPose pose;
    assert(solveControllerArm(m, input, pose, &state));
    assert(state.valid && state.side == side && !pose.wristClamped);
    Vec3 const elbow = position(pose.jointWorlds[joint(m, pose, "Elbow")]);
    Vec3 const wrist = position(pose.jointWorlds[joint(m, pose, "Wrist")]);
    assert(near(wrist, input.grip.translation, 3e-4F));
    assert(std::fabs(length(elbow-shoulder)-upper) < 3e-4F);
    assert(std::fabs(length(wrist-elbow)-0.24F) < 3e-4F);
    Vec3 const localElbow = rotate(conjugate(input.placement.rotation), elbow-shoulder);
    if (i > 0) { assert(length(localElbow-previousLocal) < 0.005F); }
    previousLocal = localElbow;
  }
  // Failures leave both the pose and state unchanged.
  auto const oldState = state;
  ControllerArmPose pose;
  auto bad = input; bad.trigger = std::numeric_limits<float>::quiet_NaN();
  assert(!solveControllerArm(m, bad, pose, &state));
  assert(state.valid == oldState.valid && state.side == oldState.side &&
         near(state.direction, oldState.direction, 1e-7F) && near(state.bend, oldState.bend, 1e-7F));
  // A reset restores the deterministic stateless result.
  state = {};
  ControllerArmPose resetPose, statelessPose;
  assert(solveControllerArm(m, input, resetPose, &state));
  assert(solveControllerArm(m, input, statelessPose));
  for (std::size_t j = 0; j < resetPose.jointWorlds.size(); ++j) {
    assert(same(resetPose.jointWorlds[j], statelessPose.jointWorlds[j]));
  }
}


void check(Model const& m, ControllerArmSide side) {
  using namespace controller_arm_detail;
  ControllerArmInput input;
  input.side = side;
  std::string const suffix = side == ControllerArmSide::Left ? "_L_" : "_R_";
  std::size_t wrist = m.nodes.size(), shoulder = m.nodes.size();
  for (std::size_t i = 0; i < m.nodes.size(); ++i) {
    if (nameIs(m.nodes[i].name, "Wrist" + suffix)) { wrist = i; }
    if (nameIs(m.nodes[i].name, "Shoulder" + suffix)) { shoulder = i; }
  }
  assert(wrist < m.nodes.size() && shoulder < m.nodes.size());
  std::vector<Transform> locals; for (auto const& n : m.nodes) { locals.push_back(n.local); }
  std::vector<Mat4> worlds(m.nodes.size()); std::vector<unsigned char> state(m.nodes.size());
  assert(globals(m, locals, wrist, worlds, state));
  Vec3 const s = position(worlds[shoulder]), w = position(worlds[wrist]);
  input.grip.translation = w;
  input.grip.rotation = rotation(worlds[wrist]);
  ControllerArmPose open;
  assert(solveControllerArm(m, input, open));
  assert(!open.wristClamped);
  assert(near(position(open.jointWorlds[joint(m, open, "Wrist")]), w));
  // At the bind pose, the IK must preserve every joint and existing skinning.
  auto const& skin = m.skins[open.skin];
  for (std::size_t i = 0; i < skin.joints.size(); ++i) {
    assert(globals(m, locals, static_cast<std::size_t>(skin.joints[i]), worlds, state));
    assert(same(open.jointWorlds[i], worlds[static_cast<std::size_t>(skin.joints[i])], 2e-4F));
  }
  std::vector<Mat4> matrices;
  assert(computeSkinMatrices(skin, Mat4{}, open.jointWorlds, matrices));
  assert(matrices.size() == skin.joints.size());
  auto triggered = input; triggered.trigger = 1;
  ControllerArmPose curled; assert(solveControllerArm(m, triggered, curled));
  assert(!same(open.jointWorlds[joint(m, open, "IndexFinger2")], curled.jointWorlds[joint(m, curled, "IndexFinger2")]));
  assert(same(open.jointWorlds[joint(m, open, "MiddleFinger2")], curled.jointWorlds[joint(m, curled, "MiddleFinger2")]));
  auto gripped = input; gripped.gripValue = 1;
  ControllerArmPose closed; assert(solveControllerArm(m, gripped, closed));
  assert(same(open.jointWorlds[joint(m, open, "IndexFinger2")], closed.jointWorlds[joint(m, closed, "IndexFinger2")]));
  assert(!same(open.jointWorlds[joint(m, open, "MiddleFinger2")], closed.jointWorlds[joint(m, closed, "MiddleFinger2")]));
  assert(!same(open.jointWorlds[joint(m, open, "ThumbFinger2")], closed.jointWorlds[joint(m, closed, "ThumbFinger2")]));
  // Curl drives the current skinning path, including mesh deformation.
  assert(computeSkinMatrices(skin, Mat4{}, closed.jointWorlds, matrices));
  MeshData mesh; mesh.vertices.resize(1); mesh.vertices[0].pos = {0.1F, 0.2F, 0.3F};
  std::vector<SkinVertexWeight> weights(1); weights[0].joints[0] = static_cast<std::uint16_t>(joint(m, closed, "MiddleFinger2")); weights[0].weights[0] = 1;
  MeshData deformed; assert(deformSkinnedMesh(mesh, weights, matrices, deformed));
  assert(near(deformed.vertices[0].pos, transformPoint(matrices[weights[0].joints[0]], mesh.vertices[0].pos)));
  auto moved = input;
  moved.grip.translation = s + (w-s)*0.8F + Vec3{0,0,0.03F};
  moved.grip.rotation = quatFromAxisAngle({0,1,0}, 0.6F);
  moved.gripToWrist.translation = {0.02F,0,0};
  ControllerArmPose posed; assert(solveControllerArm(m, moved, posed));
  assert(!posed.wristClamped);
  auto const target = moved.grip.translation + rotate(moved.grip.rotation, moved.gripToWrist.translation);
  assert(near(position(posed.jointWorlds[joint(m, posed, "Wrist")]), target, 3e-4F));
  assert(near(transformVector(posed.jointWorlds[joint(m, posed, "Wrist")], {1,0,0}), rotate(moved.grip.rotation, {1,0,0}), 3e-4F));
  moved.grip.translation = s + Vec3{100,0,0};
  assert(solveControllerArm(m, moved, posed) && posed.wristClamped);
  assert(length(position(posed.jointWorlds[joint(m, posed, "Wrist")])-s) < 2.0F);
  moved.grip.translation = s; moved.gripToWrist.translation = {};
  assert(solveControllerArm(m, moved, posed) && posed.wristClamped);
  auto placed = input;
  placed.placement.translation = {0.4F, -0.2F, 0.3F};
  placed.placement.rotation = quatFromAxisAngle({0,1,0}, 0.7F);
  placed.grip.translation = placed.placement.translation + rotate(placed.placement.rotation, w);
  placed.grip.rotation = placed.placement.rotation * input.grip.rotation;
  ControllerArmPose transformed; assert(solveControllerArm(m, placed, transformed));
  auto const placeMatrix = toMatrix(placed.placement);
  for (std::size_t i = 0; i < open.jointWorlds.size(); ++i) {
    assert(same(transformed.jointWorlds[i], mul(placeMatrix, open.jointWorlds[i]), 5e-4F));
  }
  auto bad = input; bad.trigger = std::numeric_limits<float>::quiet_NaN();
  auto previous = posed; assert(!solveControllerArm(m, bad, posed));
  assert(posed.skin == previous.skin && same(posed.jointWorlds[0], previous.jointWorlds[0]));
  bad = input; bad.grip.rotation = {0,0,0,0}; assert(!solveControllerArm(m, bad, posed));
  bad = input; bad.placement.scale.x = 2; assert(!solveControllerArm(m, bad, posed));
}
} // namespace
int main(int argc, char**) {
  auto m = fixture();
  check(m, ControllerArmSide::Left); check(m, ControllerArmSide::Right);
  Transform placement;
  for (auto side : {ControllerArmSide::Left, ControllerArmSide::Right}) {
    trajectory(m, side, placement, false);
    trajectory(m, side, placement, true);
    placement.translation = {0.4F, -0.2F, 0.3F};
    placement.rotation = quatFromAxisAngle(normalized(Vec3{1,2,3}), 0.9F);
    trajectory(m, side, placement, false);
    trajectory(m, side, placement, true);
    for (int path = 0; path < 4; ++path) {
      antipodeTrajectory(m, side, path, false);
      antipodeTrajectory(m, side, path, true);
    }
    placement = {};
  }
  auto broken = m; broken.nodes[1].name = "missing";
  ControllerArmPose out; ControllerArmInput input; input.side = ControllerArmSide::Left;
  assert(!solveControllerArm(broken, input, out));
  broken = m; broken.nodes[1].parent = 1; assert(!solveControllerArm(broken, input, out));
  if (argc > 1) {
    std::vector<std::uint8_t> bytes;
    for (int ch = std::getchar(); ch != EOF; ch = std::getchar()) { bytes.push_back(static_cast<std::uint8_t>(ch)); }
    std::string error; Model asset;
    if (!loadGlb(bytes, asset, error, false)) { std::fprintf(stderr, "asset load: %s\n", error.c_str()); return 1; }
    check(asset, ControllerArmSide::Left); check(asset, ControllerArmSide::Right);
    std::puts("real asset rig passed");
  }
  std::puts("controller arm rig tests passed");
}
