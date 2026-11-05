#pragma once

#include "vecmath.h"
#include "picking.h"

#include <initializer_list>
#include <map>
#include <algorithm>
#include <vector>

namespace vredit {

struct HandPose {
  Vec3 position{};
  Quat orientation{};
  bool valid{false};
};

// The centre is deliberately inert unless an exact preview surface was hit.
// Intersect the ring-facing plane so hand-ray origin does not skew side zones.
inline int carouselPinchDirection(Ray const& ray, Vec3 origin, Vec3 forward) {
  forward.y = 0;
  forward = normalized(forward);
  auto const denominator = dot(ray.direction, forward);
  if (!(denominator > .0001F)) { return 0; }
  float const distance = dot(origin + forward * 1.25F - ray.origin, forward) / denominator;
  if (!(distance > 0)) { return 0; }
  float const side = dot(ray.origin + ray.direction * distance - origin, cross(forward, Vec3{0,1,0}));
  return side < -.22F ? -1 : side > .22F ? 1 : 0;
}

class HandGesture {
 public:
  // Acquisition (including after tracking loss) must observe a released pinch.
  bool update(bool valid, float pinch) {
    if (!valid || !std::isfinite(pinch)) {
      reset();
      return false;
    }
    if (pinch <= 0.4F) {
      armed_ = true;
      pressed_ = false;
    } else if (armed_ && !pressed_ && pinch >= 0.55F) {
      pressed_ = true;
      return true;
    }
    return false;
  }

  bool pressed() const { return pressed_; }

  void reset() {
    armed_ = false;
    pressed_ = false;
  }

 private:
  bool armed_{false};
  bool pressed_{false};
};

// Aim availability is independent of pinch/palm tracking. Never infer a
// held pinch from an inactive or non-finite action sample.
inline bool handPinchSignal(bool handActive, bool pinchActive, float pinch) {
  return handActive && pinchActive && std::isfinite(pinch);
}
inline bool handPinchHeld(bool signal, float pinch) {
  return signal && std::isfinite(pinch) && pinch > .4F;
}

// A selection or UI interaction owns the entire overlapping squeeze. Both
// actions must return to release before a new physical grasp can be acquired.
class HandGraspGate {
 public:
  void update(bool valid, bool graspHeld, bool pinchHeld, bool captured) {
    if (!valid || captured) { blocked_ = true; }
    else if (!graspHeld && !pinchHeld) { blocked_ = false; }
  }
  void reset() { blocked_ = true; }
  bool canAcquire(bool graspPressed, bool selectionPrimed) const {
    return !blocked_ && graspPressed && selectionPrimed;
  }
 private:
  bool blocked_{true};
};

// A recent exact hit can survive a brief aim wobble on the second pinch.
// The caller verifies that the cached node is still selected and gives a
// current hit on a different object priority over the cached hit.
class HandSelectionLatch {
 public:
  static constexpr float hitGraceSeconds = .15F;
  bool startsGrab(bool hitSelection, bool recentSelection = false) const {
    return primed_ && (hitSelection || recentSelection);
  }
  void selected(bool hasSelection) { primed_ = hasSelection; clearHit(); }
  bool primed() const { return primed_; }
  void rememberHit() { hitAge_ = 0.0F; }
  void advance(float dt) {
    if (!(dt >= 0.0F) || !std::isfinite(dt)) { clearHit(); return; }
    hitAge_ += dt;
  }
  bool recentHit() const { return primed_ && hitAge_ <= hitGraceSeconds; }
  void clearHit() { hitAge_ = hitGraceSeconds + 1.0F; }
  void reset() { primed_ = false; clearHit(); }
 private:
  bool primed_{false};
  float hitAge_{hitGraceSeconds + 1.0F};
};

// Exponential low-pass filtering preserves sub-millimetre motion rather than
// quantising it. Quaternion hemisphere correction prevents sign flips.
inline HandPose smoothHandPose(HandPose previous, HandPose sample, float dt) {
  if (!previous.valid || !sample.valid) { return sample; }
  float const alpha = 1.0F - std::exp(-std::fmax(0.0F, dt) / .045F);
  auto q = sample.orientation;
  auto const& p = previous.orientation;
  if (p.x*q.x + p.y*q.y + p.z*q.z + p.w*q.w < 0) { q = {-q.x,-q.y,-q.z,-q.w}; }
  return {previous.position + (sample.position - previous.position) * alpha,
          normalized(Quat{p.x + (q.x-p.x)*alpha, p.y + (q.y-p.y)*alpha,
                          p.z + (q.z-p.z)*alpha, p.w + (q.w-p.w)*alpha}), true};
}

inline float handScaleRatio(float initialDistance, float distance) {
  if (!std::isfinite(initialDistance) || !std::isfinite(distance) || initialDistance < .04F) {
    return 1.0F;
  }
  return std::fmax(.05F, std::fmin(20.0F, distance / initialDistance));
}

// Exact nearest-triangle contact. Bounds only prune; they are never a mesh
// contact target. The local BVH is immutable and shared by mesh instances.
inline bool finiteHandPoint(Vec3 p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

inline Vec3 closestTrianglePoint(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
  auto const ab = b-a, ac = c-a, ap = p-a;
  float const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) { return a; }
  auto const bp = p-b;
  float const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) { return b; }
  float const vc = d1*d4-d3*d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { return a + ab*(d1/(d1-d3)); }
  auto const cp = p-c;
  float const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) { return c; }
  float const vb = d5*d2-d1*d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { return a + ac*(d2/(d2-d6)); }
  float const va = d3*d6-d5*d4;
  if (va <= 0 && d4-d3 >= 0 && d5-d6 >= 0) {
    return b + (c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
  }
  float const denom = va+vb+vc;
  if (!(std::fabs(denom) > 1e-20F)) { return a; }
  return a + ab*(vb/denom) + ac*(vc/denom);
}

inline float handBoxDistanceSquared(Vec3 p, Aabb box) {
  float const dx = std::fmax(0.0F, std::fmax(box.min.x-p.x, p.x-box.max.x));
  float const dy = std::fmax(0.0F, std::fmax(box.min.y-p.y, p.y-box.max.y));
  float const dz = std::fmax(0.0F, std::fmax(box.min.z-p.z, p.z-box.max.z));
  return dx*dx+dy*dy+dz*dz;
}

class HandContactMesh {
 public:
  explicit HandContactMesh(std::shared_ptr<const PickGeometry> geometry) : geometry_(std::move(geometry)) {
    if (!geometry_) { return; }
    for (std::size_t i = 0; i+2 < geometry_->indices.size(); i += 3) {
      auto const a = geometry_->indices[i], b = geometry_->indices[i+1], c = geometry_->indices[i+2];
      if (a >= geometry_->positions.size() || b >= geometry_->positions.size() || c >= geometry_->positions.size()) { continue; }
      auto const p = geometry_->positions[a], q = geometry_->positions[b], r = geometry_->positions[c];
      if (!finiteHandPoint(p) || !finiteHandPoint(q) || !finiteHandPoint(r) ||
          !(dot(cross(q-p,r-p), cross(q-p,r-p)) > 1e-20F)) { continue; }
      triangles_.push_back(i);
    }
    if (!triangles_.empty()) { build(0, triangles_.size()); }
  }

  // maxDistance is in world metres; inverse norm conservatively handles
  // nonuniform scaling, while final distances use world-space triangles.
  bool near(Vec3 point, Mat4 world, float maxDistance) const {
    Mat4 inverse;
    if (nodes_.empty() || !finiteHandPoint(point) || !std::isfinite(maxDistance) ||
        !(maxDistance >= 0) || !invertAffine(world, inverse)) { return false; }
    auto const local = transformPoint(inverse, point);
    float norm2 = 0;
    for (int col = 0; col < 3; ++col) {
      for (int row = 0; row < 3; ++row) { norm2 += inverse.m[col*4+row]*inverse.m[col*4+row]; }
    }
    float const radius2 = maxDistance*maxDistance*norm2;
    if (!finiteHandPoint(local) || !std::isfinite(radius2)) { return false; }
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
      auto const index = stack.back(); stack.pop_back();
      auto const& node = nodes_[index];
      if (handBoxDistanceSquared(local, node.bounds) > radius2) { continue; }
      if (node.count == 0) { stack.push_back(node.left); stack.push_back(node.right); continue; }
      for (std::size_t i = node.begin; i < node.begin+node.count; ++i) {
        auto const offset = triangles_[i];
        auto const a = transformPoint(world, geometry_->positions[geometry_->indices[offset]]);
        auto const b = transformPoint(world, geometry_->positions[geometry_->indices[offset+1]]);
        auto const c = transformPoint(world, geometry_->positions[geometry_->indices[offset+2]]);
        if (!finiteHandPoint(a) || !finiteHandPoint(b) || !finiteHandPoint(c)) { continue; }
        auto const normal = cross(b-a, c-a);
        if (!(dot(normal, normal) > 1e-20F)) { continue; }
        auto const delta = point-closestTrianglePoint(point,a,b,c);
        if (dot(delta,delta) <= maxDistance*maxDistance) { return true; }
      }
    }
    return false;
  }
 private:
  struct Node { Aabb bounds{}; std::size_t begin{}, count{}, left{}, right{}; };
  std::shared_ptr<const PickGeometry> geometry_;
  std::vector<std::size_t> triangles_;
  std::vector<Node> nodes_;
  Aabb triangleBounds(std::size_t offset) const {
    auto const a = geometry_->positions[geometry_->indices[offset]];
    auto const b = geometry_->positions[geometry_->indices[offset+1]];
    auto const c = geometry_->positions[geometry_->indices[offset+2]];
    return {{std::fmin(a.x,std::fmin(b.x,c.x)),std::fmin(a.y,std::fmin(b.y,c.y)),std::fmin(a.z,std::fmin(b.z,c.z))},
            {std::fmax(a.x,std::fmax(b.x,c.x)),std::fmax(a.y,std::fmax(b.y,c.y)),std::fmax(a.z,std::fmax(b.z,c.z))}};
  }
  std::size_t build(std::size_t begin, std::size_t end) {
    auto bounds = triangleBounds(triangles_[begin]);
    for (auto i = begin+1; i < end; ++i) {
      auto const b = triangleBounds(triangles_[i]);
      bounds.min = {std::fmin(bounds.min.x,b.min.x),std::fmin(bounds.min.y,b.min.y),std::fmin(bounds.min.z,b.min.z)};
      bounds.max = {std::fmax(bounds.max.x,b.max.x),std::fmax(bounds.max.y,b.max.y),std::fmax(bounds.max.z,b.max.z)};
    }
    auto const index = nodes_.size(); nodes_.push_back({bounds,begin,end-begin,0,0});
    if (end-begin > 8) {
      auto const size = bounds.max-bounds.min;
      int const axis = size.x >= size.y && size.x >= size.z ? 0 : size.y >= size.z ? 1 : 2;
      auto centre = [this,axis](std::size_t t) {
        auto const b = triangleBounds(t);
        return axis == 0 ? b.min.x+b.max.x : axis == 1 ? b.min.y+b.max.y : b.min.z+b.max.z;
      };
      auto const mid = begin+(end-begin)/2;
      std::nth_element(triangles_.begin()+begin, triangles_.begin()+mid, triangles_.begin()+end,
                       [&](auto a, auto b) { return centre(a) < centre(b); });
      auto const left = build(begin,mid), right = build(mid,end);
      nodes_[index].left = left; nodes_[index].right = right; nodes_[index].count = 0;
    }
    return index;
  }
};

// The key and the mesh each own one geometry reference. Prune only when
// no scene, undo snapshot, or other client still owns that geometry.
class HandContactCache {
 public:
  std::shared_ptr<const HandContactMesh> get(std::shared_ptr<const PickGeometry> const& geometry) {
    auto& mesh = entries_[geometry];
    if (!mesh) { mesh = std::make_shared<HandContactMesh>(geometry); }
    return mesh;
  }
  void prune() {
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (it->first.use_count() <= 2 && it->second.use_count() == 1) { it = entries_.erase(it); }
      else { ++it; }
    }
  }
  std::size_t size() const { return entries_.size(); }
 private:
  std::map<std::shared_ptr<const PickGeometry>, std::shared_ptr<const HandContactMesh>,
           std::owner_less<std::shared_ptr<const PickGeometry>>> entries_;
};

// Panel-local coordinates, with the origin at the top centre. Invalid indices
// return the origin (which is not inside any button).
inline constexpr Vec3 handPanelButtonCentre(int index) {
  int const slot = index == 0 ? 0 : index >= 8 && index <= 11 ? index - 7 : -1;
  if (slot < 0) { return {}; }
  return {(slot % 2 == 0) ? -0.115F : 0.115F,
          -0.065F * (static_cast<float>(slot / 2) + 0.5F), 0.0F};
}

struct HandPanel {
  Vec3 origin{};
  Quat orientation{};
  bool visible{false};

  int hit(Ray const& ray) const {
    if (!visible) {
      return -1;
    }
    auto const normal = rotate(orientation, {0.0F, 0.0F, 1.0F});
    auto const denominator = dot(ray.direction, normal);
    if (!(denominator < -1e-6F)) {
      return -1;
    }
    auto const distance = dot(origin - ray.origin, normal) / denominator;
    if (!(distance >= 0.0F) || !std::isfinite(distance)) {
      return -1;
    }
    auto const local = rotate(conjugate(orientation),
                              ray.origin + ray.direction * distance - origin);
    for (int index : {0, 8, 9, 10, 11}) {
      auto const centre = handPanelButtonCentre(index);
      if (std::fabs(local.x - centre.x) <= 0.105F &&
          std::fabs(local.y - centre.y) <= 0.028F) {
        return index;
      }
    }
    return -1;
  }
};

inline bool palmFacingUp(HandPose const& pose, bool previous) {
  return pose.valid &&
         rotate(pose.orientation, {0.0F, -1.0F, 0.0F}).y > (previous ? 0.55F : 0.75F);
}

// Frame latched at grab start. Projection offsets are relative to the eye,
// so a remote surface follows the same fraction of the view as the hand.
struct HandGrabView {
  Vec3 eye{};
  Quat orientation{}; // local -Z is forward
  bool valid{false};
};

inline Vec3 projectedGrabPoint(HandGrabView const& view, Vec3 handStart, Vec3 handNow, Vec3 point) {
  if (!view.valid) { return point + (handNow - handStart); }
  auto const inv = conjugate(view.orientation);
  auto const h0 = rotate(inv, handStart - view.eye);
  auto const h1 = rotate(inv, handNow - view.eye);
  auto const p = rotate(inv, point - view.eye);
  float const z0 = -h0.z, z1 = -h1.z, zp = -p.z;
  if (!(z0 > .05F && zp > .05F)) { return point + (handNow - handStart); }
  // Bound gain for a hand starting close to or behind the eye plane. At normal
  // hand distances the surface follows the hand's exact fractional depth.
  float const startDepth = std::fmax(.10F, z0);
  float const currentDepth = std::fmax(.10F, z1);
  float const depth = std::fmax(.05F, zp * (1.0F + (z1 - z0) / startDepth));
  // This is the difference of projected coordinates, attenuated smoothly
  // near the eye plane. It remains zero for motion along the same eye ray.
  float const denom = startDepth * currentDepth;
  Vec3 const target{(p.x / zp + (h1.x * z0 - h0.x * z1) / denom) * depth,
                    (p.y / zp + (h1.y * z0 - h0.y * z1) / denom) * depth, -depth};
  return view.eye + rotate(view.orientation, target);
}

inline Mat4 projectedHandGrabTransform(HandGrabView const& view, HandPose start,
                                      HandPose current, Vec3 pivot, Mat4 initialWorld) {
  if (!start.valid || !current.valid) { return initialWorld; }
  if (current.position.x == start.position.x && current.position.y == start.position.y &&
      current.position.z == start.position.z && start.orientation.x == current.orientation.x &&
      start.orientation.y == current.orientation.y && start.orientation.z == current.orientation.z &&
      start.orientation.w == current.orientation.w) { return initialWorld; }
  auto const rotation = normalized(current.orientation * conjugate(start.orientation));
  auto const target = projectedGrabPoint(view, start.position, current.position, pivot);
  return mul(translationMatrix(target), mul(rotationMatrix(rotation),
             mul(translationMatrix(-pivot), initialWorld)));
}

// Poses/orientations are world-space unit quaternions, as supplied by tracking.
// Always reapply the cumulative delta to the captured initialWorld, never the
// previous result. The caller converts world to parent-local if needed.
inline Mat4 handGrabTransform(HandPose start, HandPose current, Vec3 pivot, Mat4 initialWorld) {
  if (!start.valid || !current.valid) {
    return initialWorld;
  }
  auto const& a = start.orientation;
  auto const& b = current.orientation;
  if (current.position.x == start.position.x && current.position.y == start.position.y &&
      current.position.z == start.position.z && a.x == b.x && a.y == b.y &&
      a.z == b.z && a.w == b.w) {
    return initialWorld;
  }
  auto const delta = normalized(current.orientation * conjugate(start.orientation));

  auto const translation = (current.position - start.position);
  return mul(translationMatrix(pivot + translation),
             mul(rotationMatrix(delta), mul(translationMatrix(-pivot), initialWorld)));
}

} // namespace vredit
