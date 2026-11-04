#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "scene.h"

namespace vredit {

struct Model;

enum class AnimationPath : std::uint8_t { Translation, Rotation, Scale };
enum class AnimationInterpolation : std::uint8_t { Step, Linear, CubicSpline };

// Components are glTF order (x, y, z, w); w is unused for translation/scale.
struct AnimationValue {
  float x{0.0F}, y{0.0F}, z{0.0F}, w{0.0F};
};

struct AnimationChannel {
  std::size_t node{0}; // Model::nodes index, including primitive-wrapper offsets
  AnimationPath path{AnimationPath::Translation};
  AnimationInterpolation interpolation{AnimationInterpolation::Linear};
  std::vector<float> times;
  // CubicSpline stores [in tangent, value, out tangent] per key.
  std::vector<AnimationValue> values;
};

struct AnimationClip {
  std::string name;
  std::vector<AnimationChannel> channels;
};

// Produces local transforms in Model::nodes order, starting from the model's
// base pose each call. Time is clamped per channel; no looping or scene writes.
// Returns false without changing out for malformed data or non-finite time.
bool evaluateAnimation(Model const& model, AnimationClip const& clip, float time,
                       std::vector<Transform>& out);

} // namespace vredit
