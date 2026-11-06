#pragma once

#include <algorithm>
#include <cmath>

#include "vecmath.h"

namespace vredit {

// A distant light: direction points from the scene toward the light. Ambient
// is a uniform fill; this deliberately is not a position-dependent point light.
struct SceneLighting {
  Vec3 direction{0.35F, 0.85F, 0.45F};
  float ambient{0.35F};
};

inline bool sameLighting(SceneLighting const& a, SceneLighting const& b) {
  return a.direction.x == b.direction.x && a.direction.y == b.direction.y &&
         a.direction.z == b.direction.z && a.ambient == b.ambient;
}

enum class LightingAdjustment { left, right, up, down, dimmer, brighter, reset };

inline void adjustLighting(SceneLighting& light, LightingAdjustment adjustment) {
  constexpr float step = 0.261799388F; // 15 degrees
  constexpr float limit = 1.48352986F; // +/-85 degrees, keeping azimuth defined
  if (adjustment == LightingAdjustment::reset) {
    light = {};
    return;
  }
  if (adjustment == LightingAdjustment::dimmer || adjustment == LightingAdjustment::brighter) {
    float const change = adjustment == LightingAdjustment::brighter ? 0.05F : -0.05F;
    light.ambient = std::clamp(light.ambient + change, 0.0F, 1.0F);
    return;
  }
  Vec3 const direction = normalized(light.direction);
  float yaw = std::atan2(direction.x, direction.z);
  float elevation = std::asin(std::clamp(direction.y, -1.0F, 1.0F));
  if (adjustment == LightingAdjustment::left) { yaw -= step; }
  if (adjustment == LightingAdjustment::right) { yaw += step; }
  if (adjustment == LightingAdjustment::up) { elevation = std::min(limit, elevation + step); }
  if (adjustment == LightingAdjustment::down) { elevation = std::max(-limit, elevation - step); }
  float const horizontal = std::cos(elevation);
  light.direction = {std::sin(yaw) * horizontal, std::sin(elevation), std::cos(yaw) * horizontal};
}

} // namespace vredit
