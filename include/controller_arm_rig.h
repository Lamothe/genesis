#pragma once

#include "mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vredit {

// Controller-driven pose for the Male Hand Rigged asset. Placement is
// the asset's bind-space-to-world transform (normally head/shoulder anchored).
// gripToWrist is a calibrated rigid offset in controller grip coordinates. Both
// use metres. The caller owns tracking validity, visibility and control actions.
enum class ControllerArmSide : std::uint8_t { Left, Right };
struct ControllerArmInput {
  ControllerArmSide side{ControllerArmSide::Right};
  Transform placement{};
  Transform grip{};
  Transform gripToWrist{};
  float trigger{0.0F};
  float gripValue{0.0F};
};

// One state per independently tracked side. Directions are stored in placement
// coordinates so head/body rotation does not leave the bend behind. Reset on
// tracking interruption or when changing rigs. Failed solves do not update it.
struct ControllerArmState {
  bool valid{false};
  ControllerArmSide side{ControllerArmSide::Right};
  Vec3 direction{};
  Vec3 bend{};
};

// jointWorlds is in the selected ModelSkin::joints order and can be passed
// directly to computeSkinMatrices with the actual skin mesh's world matrix.
// wristClamped reports targets outside the two-bone reachable annulus.
struct ControllerArmPose {
  std::size_t skin{0};
  std::vector<Mat4> jointWorlds;
  bool wristClamped{false};
};

namespace controller_arm_detail {
inline bool finite(Vec3 v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
inline bool validRotation(Quat q) {
  float const n = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
  return std::isfinite(n) && n > 1e-12F;
}
inline bool rigid(Transform const& t) {
  return finite(t.translation) && validRotation(t.rotation) &&
         std::isfinite(t.scale.x) && std::isfinite(t.scale.y) && std::isfinite(t.scale.z) &&
         std::fabs(t.scale.x - 1.0F) < 1e-5F &&
         std::fabs(t.scale.y - 1.0F) < 1e-5F && std::fabs(t.scale.z - 1.0F) < 1e-5F;
}
inline Vec3 pos(Mat4 const& m) { return {m.m[12], m.m[13], m.m[14]}; }
inline Quat rotation(Mat4 const& m) {
  Vec3 t, s;
  Quat r;
  decompose(m, t, r, s);
  return r;
}
inline Quat fromTo(Vec3 a, Vec3 b) {
  a = normalized(a); b = normalized(b);
  float const d = std::clamp(dot(a, b), -1.0F, 1.0F);
  Vec3 const c = cross(a, b);
  float const sine = length(c);
  if (sine > 1e-7F) {
    return quatFromAxisAngle(c / sine, std::atan2(sine, d));
  }
  if (d < 0.0F) {
    Vec3 const axis = cross(a, std::fabs(a.x) < 0.8F ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
    return quatFromAxisAngle(axis, 3.14159265358979323846F);
  }
  return Quat{};
}
inline Mat4 around(Vec3 p, Quat q) {
  return mul(translationMatrix(p), mul(rotationMatrix(q), translationMatrix(-p)));
}
inline bool nameIs(std::string const& name, std::string const& stem) {
  if (name.size() <= stem.size() || name.compare(0, stem.size(), stem) != 0) {
    return false;
  }
  // Exporter appends a numeric identifier; do not match FK/control bones.
  for (std::size_t i = stem.size(); i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') { return false; }
  }
  return true;
}
inline bool descendant(Model const& model, std::size_t node, std::size_t ancestor) {
  for (std::size_t count = 0; count < model.nodes.size(); ++count) {
    if (node == ancestor) { return true; }
    auto const p = model.nodes[node].parent;
    if (p < 0 || static_cast<std::size_t>(p) >= model.nodes.size()) { return false; }
    node = static_cast<std::size_t>(p);
  }
  return false;
}
inline bool globals(Model const& model, std::vector<Transform> const& locals,
                    std::size_t index, std::vector<Mat4>& world, std::vector<unsigned char>& state) {
  if (state[index] == 2) { return true; }
  if (state[index] == 1) { return false; }
  state[index] = 1;
  auto const& t = locals[index];
  if (!finite(t.translation) || !finite(t.scale) || !validRotation(t.rotation) ||
      std::fabs(t.scale.x) < 1e-8F || std::fabs(t.scale.y) < 1e-8F || std::fabs(t.scale.z) < 1e-8F) { return false; }
  auto const parent = model.nodes[index].parent;
  if (parent < -1 || (parent >= 0 && static_cast<std::size_t>(parent) >= model.nodes.size())) { return false; }
  Mat4 const local = trs(t.translation, normalized(t.rotation), t.scale);
  if (parent >= 0) {
    auto const p = static_cast<std::size_t>(parent);
    if (!globals(model, locals, p, world, state)) { return false; }
    world[index] = mul(world[p], local);
  } else { world[index] = local; }
  for (float value : world[index].m) { if (!std::isfinite(value)) { return false; } }
  state[index] = 2;
  return true;
}
} // namespace controller_arm_detail

// Failure leaves out untouched. Model nodes are never modified. Skin matching
// uses the actual deform bones (Shoulder/Elbow/Wrist and Finger1..3), not the
// similarly named FK controls. Independent calls can pose either arm, even
// when the other controller is untracked. The bind bend is transported with
// the wrist direction; this is approximate IK, not measured body tracking.
// If supplied, state parallel-transports the previous bend between samples;
// it is path-dependent and cannot resolve an instantaneous antipodal jump.
inline bool solveControllerArm(Model const& model, ControllerArmInput const& input,
                               ControllerArmPose& out, ControllerArmState* continuity = nullptr) {
  using namespace controller_arm_detail;
  if (!rigid(input.placement) || !rigid(input.grip) || !rigid(input.gripToWrist) ||
      !std::isfinite(input.trigger) || !std::isfinite(input.gripValue)) { return false; }
  char const* side = input.side == ControllerArmSide::Left ? "_L_" : "_R_";
  if (input.side != ControllerArmSide::Left && input.side != ControllerArmSide::Right) { return false; }
  auto find = [&](std::string const& stem) -> std::int32_t {
    std::int32_t found = -1;
    for (std::size_t i = 0; i < model.nodes.size(); ++i) {
      if (nameIs(model.nodes[i].name, stem + side)) {
        if (found >= 0 || i > static_cast<std::size_t>(INT32_MAX)) { return -1; }
        found = static_cast<std::int32_t>(i);
      }
    }
    return found;
  };
  auto const shoulder = find("Shoulder"), elbow = find("Elbow"), wrist = find("Wrist");
  if (shoulder < 0 || elbow < 0 || wrist < 0) { return false; }
  auto const sh = static_cast<std::size_t>(shoulder), el = static_cast<std::size_t>(elbow), wr = static_cast<std::size_t>(wrist);
  if (!descendant(model, el, sh) || !descendant(model, wr, el)) { return false; }
  std::array<std::array<std::size_t, 3>, 5> fingers{};
  char const* const names[] = {"IndexFinger", "MiddleFinger", "RingFinger", "PinkyFinger", "ThumbFinger"};
  for (std::size_t f = 0; f < fingers.size(); ++f) {
    for (std::size_t j = 0; j < 3; ++j) {
      auto const n = find(std::string(names[f]) + std::to_string(j + 1));
      if (n < 0) { return false; }
      fingers[f][j] = static_cast<std::size_t>(n);
      if (!descendant(model, fingers[f][j], j == 0 ? wr : fingers[f][j-1]) ||
          fingers[f][j] == (j == 0 ? wr : fingers[f][j-1])) { return false; }
    }
  }
  std::size_t skin = model.skins.size();
  for (std::size_t s = 0; s < model.skins.size(); ++s) {
    auto const& joints = model.skins[s].joints;
    if (std::find(joints.begin(), joints.end(), wrist) != joints.end()) {
      if (skin != model.skins.size()) { return false; }
      skin = s;
    }
  }
  if (skin == model.skins.size()) { return false; }
  auto const& selected = model.skins[skin];
  if (selected.joints.empty() || selected.inverseBindMatrices.size() != selected.joints.size()) { return false; }
  auto hasJoint = [&](std::size_t n) { return std::find(selected.joints.begin(), selected.joints.end(), static_cast<std::int32_t>(n)) != selected.joints.end(); };
  if (!hasJoint(sh) || !hasJoint(el)) { return false; }
  for (auto const& finger : fingers) { for (auto n : finger) { if (!hasJoint(n)) { return false; } } }
  std::vector<Transform> locals;
  locals.reserve(model.nodes.size());
  for (auto const& node : model.nodes) { locals.push_back(node.local); }
  float const trigger = std::clamp(input.trigger, 0.0F, 1.0F);
  float const grip = std::clamp(input.gripValue, 0.0F, 1.0F);
  for (std::size_t f = 0; f < fingers.size(); ++f) {
    float const amount = f == 0 ? trigger : (f == 4 ? grip * 0.55F : grip);
    constexpr float angles[] = {0.65F, 0.9F, 0.65F};
    for (std::size_t j = 0; j < 3; ++j) {
      auto& r = locals[fingers[f][j]].rotation;
      r = normalized(r * quatFromAxisAngle({0, 1, 0}, amount * angles[j]));
    }
  }
  std::vector<Mat4> world(model.nodes.size());
  std::vector<unsigned char> state(model.nodes.size());
  // Only selected skin joints and ancestors matter: unrelated exporter control
  // nodes cannot invalidate a usable skin.
  for (auto j : selected.joints) {
    if (j < 0 || static_cast<std::size_t>(j) >= model.nodes.size() ||
        !globals(model, locals, static_cast<std::size_t>(j), world, state)) { return false; }
  }
  Mat4 const placement = trs(input.placement.translation, normalized(input.placement.rotation), {1,1,1});
  for (std::size_t i = 0; i < world.size(); ++i) { if (state[i] == 2) { world[i] = mul(placement, world[i]); } }
  Mat4 const target = mul(trs(input.grip.translation, normalized(input.grip.rotation), {1,1,1}),
                          trs(input.gripToWrist.translation, normalized(input.gripToWrist.rotation), {1,1,1}));
  Vec3 const s = pos(world[sh]), e = pos(world[el]), w = pos(world[wr]);
  float const upper = length(e-s), lower = length(w-e);
  Vec3 const targetDelta = pos(target)-s;
  float const distance = length(targetDelta);
  if (!std::isfinite(upper) || !std::isfinite(lower) || !std::isfinite(distance) || upper < 1e-5F || lower < 1e-5F) { return false; }
  float const minReach = std::fabs(upper-lower) + 1e-6F;
  float const maxReach = upper + lower - 1e-6F;
  if (minReach >= maxReach) { return false; }
  float const reach = std::clamp(distance, minReach, maxReach);
  Vec3 const direction = distance > 1e-7F ? targetDelta / distance : normalized(w-s);
  if (length(direction) < 0.5F) { return false; }
  // Transport the bind bend from the bind shoulder-to-wrist direction.
  // Projecting the upper arm directly onto the target plane reverses its
  // sign as the target crosses the bind upper arm, flipping the elbow.
  Vec3 const bindDelta = w-s;
  float const bindDistance = length(bindDelta);
  Vec3 const bindDirection = bindDistance > 1e-7F ? bindDelta / bindDistance : normalized(e-s);
  Vec3 bindBend = (e-s) - bindDirection * dot(e-s, bindDirection);
  if (length(bindBend) < 1e-6F) {
    // A straight bind arm has no bend plane: prefer down in placement space.
    Vec3 axis = rotate(normalized(input.placement.rotation), {0,-1,0});
    if (length(cross(bindDirection, axis)) < 1e-4F) {
      axis = rotate(normalized(input.placement.rotation), {0,0,1});
    }
    bindBend = axis - bindDirection * dot(axis, bindDirection);
  }
  bindBend = normalized(bindBend);
  float const alignment = std::clamp(dot(bindDirection, direction), -1.0F, 1.0F);
  Vec3 const axis = cross(bindDirection, direction);
  float const sine = length(axis);
  Vec3 bend;
  if (sine > 1e-7F) {
    // Shortest-arc transport; atan2 remains stable close to the antipode.
    bend = rotate(quatFromAxisAngle(axis / sine, std::atan2(sine, alignment)), bindBend);
  } else {
    // The parallel and antipodal directions need no well-defined axis.
    // At the antipode there is no unique shortest arc. Keep the bind pole
    // as the deterministic limiting choice; a stateless solver cannot be
    // continuous for all paths through this point.
    bend = bindBend - direction * dot(bindBend, direction);
  }
  bend = normalized(bend);
  Quat const placementRotation = normalized(input.placement.rotation);
  if (continuity != nullptr && continuity->valid && continuity->side == input.side) {
    Vec3 const previousDirection = rotate(placementRotation, continuity->direction);
    Vec3 const previousBend = rotate(placementRotation, continuity->bend);
    Vec3 const stepAxis = cross(previousDirection, direction);
    float const stepSine = length(stepAxis);
    float const stepCosine = std::clamp(dot(previousDirection, direction), -1.0F, 1.0F);
    if (stepSine > 1e-7F) {
      bend = rotate(quatFromAxisAngle(stepAxis / stepSine, std::atan2(stepSine, stepCosine)), previousBend);
    } else {
      // At an instantaneous 180-degree jump, the previous bend itself is
      // a valid rotation axis; there is no uniquely determined path.
      bend = previousBend;
    }
    bend = normalized(bend - direction * dot(bend, direction));
    if (!finite(bend) || length(bend) < 0.5F) { return false; }
  }
  float const along = (upper*upper + reach*reach - lower*lower)/(2.0F*reach);
  float const height = std::sqrt(std::max(0.0F, upper*upper-along*along));
  Vec3 const desiredElbow = s + direction*along + bend*height;
  Vec3 const desiredWrist = s + direction*reach;
  Mat4 const shoulderDelta = around(s, fromTo(e-s, desiredElbow-s));
  auto apply = [&](std::size_t ancestor, Mat4 const& delta) {
    for (std::size_t i = 0; i < world.size(); ++i) {
      if (state[i] == 2 && descendant(model, i, ancestor)) { world[i] = mul(delta, world[i]); }
    }
  };
  apply(sh, shoulderDelta);
  apply(el, around(pos(world[el]), fromTo(pos(world[wr])-pos(world[el]), desiredWrist-pos(world[el]))));
  Quat const correction = normalized(rotation(target) * conjugate(rotation(world[wr])));
  apply(wr, around(pos(world[wr]), correction));
  ControllerArmPose result;
  result.skin = skin;
  result.wristClamped = distance < minReach || distance > maxReach;
  result.jointWorlds.reserve(selected.joints.size());
  for (auto j : selected.joints) {
    Mat4 const& m = world[static_cast<std::size_t>(j)];
    for (float value : m.m) { if (!std::isfinite(value)) { return false; } }
    result.jointWorlds.push_back(m);
  }
  out = std::move(result);
  if (continuity != nullptr) {
    continuity->valid = true;
    continuity->side = input.side;
    continuity->direction = rotate(conjugate(placementRotation), direction);
    continuity->bend = rotate(conjugate(placementRotation), bend);
  }
  return true;
}

} // namespace vredit
