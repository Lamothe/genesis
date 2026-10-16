#include "camera.h"

#include <cmath>

namespace vredit {

namespace {

constexpr float kPi = 3.14159265359F;
constexpr float kMaxPitch = 1.55F;
constexpr float kMinDistance = 0.05F;
constexpr float kMaxDistance = 500.0F;

} // namespace

Vec3 cameraPosition(Camera const& camera) {
  float cp = std::cos(camera.pitch);
  float sp = std::sin(camera.pitch);
  float sy = std::sin(camera.yaw);
  float cy = std::cos(camera.yaw);
  return camera.target + Vec3{cp * sy, sp, cp * cy} * camera.distance;
}

Vec3 cameraForward(Camera const& camera) {
  return normalized(camera.target - cameraPosition(camera));
}

Vec3 cameraRight(Camera const& camera) {
  return normalized(cross(cameraForward(camera), Vec3{0.0F, 1.0F, 0.0F}));
}

Vec3 cameraUp(Camera const& camera) {
  return cross(cameraRight(camera), cameraForward(camera));
}

Mat4 cameraView(Camera const& camera) {
  Vec3 eye = cameraPosition(camera);
  Vec3 f = cameraForward(camera);
  Vec3 r = cameraRight(camera);
  Vec3 u = cameraUp(camera);
  Mat4 view;
  view.m[0] = r.x;
  view.m[4] = r.y;
  view.m[8] = r.z;
  view.m[12] = -dot(r, eye);
  view.m[1] = u.x;
  view.m[5] = u.y;
  view.m[9] = u.z;
  view.m[13] = -dot(u, eye);
  view.m[2] = -f.x;
  view.m[6] = -f.y;
  view.m[10] = -f.z;
  view.m[14] = dot(f, eye);
  view.m[3] = 0.0F;
  view.m[7] = 0.0F;
  view.m[11] = 0.0F;
  view.m[15] = 1.0F;
  return view;
}

Mat4 cameraProjection(Camera const& camera, float aspect) {
  float f = 1.0F / std::tan(camera.fovY * 0.5F);
  Mat4 proj; // zero-initialised below
  for (float& element : proj.m) {
    element = 0.0F;
  }
  proj.m[0] = f / aspect;
  proj.m[5] = -f; // clip Y grows downward in Vulkan
  proj.m[10] = camera.farZ / (camera.nearZ - camera.farZ);
  proj.m[11] = -1.0F;
  proj.m[14] = (camera.nearZ * camera.farZ) / (camera.nearZ - camera.farZ);
  return proj;
}

Mat4 cameraViewProjection(Camera const& camera, float aspect) {
  return mul(cameraProjection(camera, aspect), cameraView(camera));
}

namespace {

// Maps NDC (x, y) at view-space distance d along the view axis to view space.
Vec3 ndcToView(Camera const& camera, float aspect, float ndcX, float ndcY, float d) {
  float f = 1.0F / std::tan(camera.fovY * 0.5F);
  return {ndcX * d * aspect / f, -ndcY * d / f, -d};
}

} // namespace

Ray cameraRay(Camera const& camera, float aspect, Vec2 pixel, Vec2 viewportSize) {
  float ndcX = (pixel.x / viewportSize.x) * 2.0F - 1.0F;
  float ndcY = (pixel.y / viewportSize.y) * 2.0F - 1.0F;
  Mat4 invView;
  if (!invertAffine(cameraView(camera), invView)) {
    return {cameraPosition(camera), cameraForward(camera)};
  }
  Vec3 nearWorld = transformPoint(invView, ndcToView(camera, aspect, ndcX, ndcY, camera.nearZ));
  Vec3 farWorld = transformPoint(invView, ndcToView(camera, aspect, ndcX, ndcY, camera.farZ));
  return {cameraPosition(camera), normalized(farWorld - nearWorld)};
}

void cameraOrbit(Camera& camera, float dYaw, float dPitch) {
  camera.yaw += dYaw;
  camera.pitch += dPitch;
  if (camera.pitch > kMaxPitch) {
    camera.pitch = kMaxPitch;
  } else if (camera.pitch < -kMaxPitch) {
    camera.pitch = -kMaxPitch;
  }
}

void cameraDolly(Camera& camera, float amount) {
  float factor = std::exp(-amount * 0.15F);
  float distance = camera.distance * factor;
  if (distance < kMinDistance) {
    distance = kMinDistance;
  } else if (distance > kMaxDistance) {
    distance = kMaxDistance;
  }
  camera.distance = distance;
}

void cameraPan(Camera& camera, float dPixelX, float dPixelY, float viewportHeightPx) {
  if (viewportHeightPx <= 0.0F) {
    return;
  }
  float worldPerPixel =
      2.0F * camera.distance * std::tan(camera.fovY * 0.5F) / viewportHeightPx;
  camera.target = camera.target - cameraRight(camera) * (dPixelX * worldPerPixel) +
                  cameraUp(camera) * (dPixelY * worldPerPixel);
}

} // namespace vredit
