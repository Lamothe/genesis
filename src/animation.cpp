#include "animation.h"

#include <algorithm>
#include <cmath>

#include "mesh.h"

namespace vredit {
namespace {

bool finite(AnimationValue const& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}

AnimationValue mix(AnimationValue a, AnimationValue b, float t) {
  return {a.x * (1.0F - t) + b.x * t, a.y * (1.0F - t) + b.y * t,
          a.z * (1.0F - t) + b.z * t, a.w * (1.0F - t) + b.w * t};
}

bool unit(AnimationValue& v) {
  double n = std::sqrt(static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y +
                       static_cast<double>(v.z) * v.z + static_cast<double>(v.w) * v.w);
  if (!(n > 0.0) || !std::isfinite(n)) {
    return false;
  }
  v = {static_cast<float>(v.x / n), static_cast<float>(v.y / n),
       static_cast<float>(v.z / n), static_cast<float>(v.w / n)};
  return true;
}

AnimationValue slerp(AnimationValue a, AnimationValue b, float t) {
  float d = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
  if (d < 0.0F) {
    b = {-b.x, -b.y, -b.z, -b.w};
    d = -d;
  }
  d = std::clamp(d, 0.0F, 1.0F);
  if (d > 0.9995F) {
    return mix(a, b, t);
  }
  float angle = std::acos(d);
  float denom = std::sin(angle);
  float x = std::sin((1.0F - t) * angle) / denom;
  float y = std::sin(t * angle) / denom;
  return {a.x*x + b.x*y, a.y*x + b.y*y, a.z*x + b.z*y, a.w*x + b.w*y};
}

AnimationValue cubic(AnimationValue a, AnimationValue out, AnimationValue b,
                     AnimationValue in, float t, float dt) {
  float t2 = t*t, t3 = t2*t;
  float h0 = 2*t3 - 3*t2 + 1, h1 = t3 - 2*t2 + t;
  float h2 = -2*t3 + 3*t2, h3 = t3 - t2;
  return {h0*a.x + h1*dt*out.x + h2*b.x + h3*dt*in.x,
          h0*a.y + h1*dt*out.y + h2*b.y + h3*dt*in.y,
          h0*a.z + h1*dt*out.z + h2*b.z + h3*dt*in.z,
          h0*a.w + h1*dt*out.w + h2*b.w + h3*dt*in.w};
}

bool sample(AnimationChannel const& c, float time, AnimationValue& value) {
  std::size_t n = c.times.size();
  bool spline = c.interpolation == AnimationInterpolation::CubicSpline;
  if (c.interpolation != AnimationInterpolation::Step &&
      c.interpolation != AnimationInterpolation::Linear && !spline) {
    return false;
  }
  if (n == 0 || (spline && n > c.values.max_size() / 3U) ||
      c.values.size() != n * (spline ? 3U : 1U)) {
    return false;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(c.times[i]) || c.times[i] < 0 ||
        (i && c.times[i] <= c.times[i-1])) {
      return false;
    }
  }
  for (std::size_t i = 0; i < c.values.size(); ++i) {
    auto v = c.values[i];
    if (!finite(v) || (c.path == AnimationPath::Rotation &&
        (!spline || i % 3U == 1U) && !unit(v))) {
      return false;
    }
  }
  auto key = [&](std::size_t i) { return c.values[spline ? i*3U + 1U : i]; };
  auto upper = std::upper_bound(c.times.begin(), c.times.end(), time);
  if (upper == c.times.begin()) {
    value = key(0);
  } else if (upper == c.times.end()) {
    value = key(n-1);
  } else {
    std::size_t b = static_cast<std::size_t>(upper - c.times.begin());
    std::size_t a = b - 1U;
    float dt = c.times[b] - c.times[a];
    float t = (time - c.times[a]) / dt;
    if (c.interpolation == AnimationInterpolation::Step) {
      value = key(a);
    } else if (spline) {
      value = cubic(key(a), c.values[a*3U+2U], key(b), c.values[b*3U], t, dt);
    } else if (c.path == AnimationPath::Rotation) {
      AnimationValue qa = key(a), qb = key(b);
      if (!unit(qa) || !unit(qb)) {
        return false;
      }
      value = slerp(qa, qb, t);
    } else {
      value = mix(key(a), key(b), t);
    }
  }
  return finite(value) && (c.path != AnimationPath::Rotation || unit(value));
}
} // namespace

bool evaluateAnimation(Model const& model, AnimationClip const& clip, float time,
                       std::vector<Transform>& out) {
  if (!std::isfinite(time)) {
    return false;
  }
  std::vector<Transform> pose;
  pose.reserve(model.nodes.size());
  for (auto const& node : model.nodes) {
    pose.push_back(node.local);
  }
  for (auto const& channel : clip.channels) {
    if (channel.node >= pose.size()) {
      return false;
    }
    AnimationValue v;
    if (!sample(channel, time, v)) {
      return false;
    }
    Transform& target = pose[channel.node];
    switch (channel.path) {
    case AnimationPath::Translation: target.translation = {v.x, v.y, v.z}; break;
    case AnimationPath::Rotation: target.rotation = {v.x, v.y, v.z, v.w}; break;
    case AnimationPath::Scale: target.scale = {v.x, v.y, v.z}; break;
    default: return false;
    }
  }
  out = std::move(pose);
  return true;
}
} // namespace vredit
