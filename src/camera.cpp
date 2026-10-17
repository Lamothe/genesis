#include "camera.h"

#include <cmath>

namespace vredit {

namespace {

constexpr float kPi = 3.14159265359F;
constexpr float kMaxPitch = 1.55F;

} // namespace

Vec3 cameraForward(Camera const& camera) {
  float cp = std::cos(camera.pitch);
  float sp = std::sin(camera.pitch);
  float sy = std::sin(camera.yaw);
  float cy = std::cos(camera.yaw);
  return {sy * cp, sp, -cy * cp};
}

Vec3 cameraRight(Camera const& camera) {
  return normalized(cross(cameraForward(camera), Vec3{0.0F, 1.0F, 0.0F}));
}

Vec3 cameraUp(Camera const& camera) {
  return cross(cameraRight(camera), cameraForward(camera));
}

Mat4 cameraView(Camera const& camera) {
  Vec3 const& eye = camera.position;
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

Mat4 cameraProjectionFromTangents(float tanLeft, float tanRight, float tanUp, float tanDown,
                                  float nearZ, float farZ) {
  Mat4 proj;
  for (float& element : proj.m) {
    element = 0.0F;
  }
  proj.m[0] = 2.0F / (tanRight - tanLeft);
  proj.m[5] = -2.0F / (tanUp - tanDown); // clip Y grows downward in Vulkan
  proj.m[8] = (tanRight + tanLeft) / (tanRight - tanLeft);
  proj.m[9] = -(tanUp + tanDown) / (tanUp - tanDown);
  proj.m[10] = farZ / (nearZ - farZ);
  proj.m[11] = -1.0F;
  proj.m[14] = (nearZ * farZ) / (nearZ - farZ);
  return proj;
}

Mat4 cameraViewFromPose(Vec3 const& position, Quat const& orientation) {
  // Orientation is world <- eye; the view needs its inverse: R^T and -R^T t.
  Quat const& q = orientation;
  float xx = q.x * q.x;
  float yy = q.y * q.y;
  float zz = q.z * q.z;
  float xy = q.x * q.y;
  float xz = q.x * q.z;
  float yz = q.y * q.z;
  float xw = q.x * q.w;
  float yw = q.y * q.w;
  float zw = q.z * q.w;
  float r00 = 1.0F - 2.0F * (yy + zz);
  float r01 = 2.0F * (xy - zw);
  float r02 = 2.0F * (xz + yw);
  float r10 = 2.0F * (xy + zw);
  float r11 = 1.0F - 2.0F * (xx + zz);
  float r12 = 2.0F * (yz - xw);
  float r20 = 2.0F * (xz - yw);
  float r21 = 2.0F * (yz + xw);
  float r22 = 1.0F - 2.0F * (xx + yy);

  Mat4 view;
  view.m[0] = r00;
  view.m[4] = r10;
  view.m[8] = r20;
  view.m[1] = r01;
  view.m[5] = r11;
  view.m[9] = r21;
  view.m[2] = r02;
  view.m[6] = r12;
  view.m[10] = r22;
  view.m[12] = -(r00 * position.x + r10 * position.y + r20 * position.z);
  view.m[13] = -(r01 * position.x + r11 * position.y + r21 * position.z);
  view.m[14] = -(r02 * position.x + r12 * position.y + r22 * position.z);
  view.m[3] = 0.0F;
  view.m[7] = 0.0F;
  view.m[11] = 0.0F;
  view.m[15] = 1.0F;
  return view;
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
    return {camera.position, cameraForward(camera)};
  }
  Vec3 nearWorld = transformPoint(invView, ndcToView(camera, aspect, ndcX, ndcY, camera.nearZ));
  Vec3 farWorld = transformPoint(invView, ndcToView(camera, aspect, ndcX, ndcY, camera.farZ));
  return {camera.position, normalized(farWorld - nearWorld)};
}

void cameraLook(Camera& camera, float dYaw, float dPitch) {
  camera.yaw += dYaw;
  camera.pitch += dPitch;
  if (camera.pitch > kMaxPitch) {
    camera.pitch = kMaxPitch;
  } else if (camera.pitch < -kMaxPitch) {
    camera.pitch = -kMaxPitch;
  }
  // Keep yaw bounded for float health on long sessions.
  if (camera.yaw > kPi || camera.yaw < -kPi) {
    camera.yaw = std::remainder(camera.yaw, 2.0F * kPi);
  }
}

void cameraFly(Camera& camera, Vec3 localMetres) {
  camera.position = camera.position + cameraRight(camera) * localMetres.x +
                    Vec3{0.0F, 1.0F, 0.0F} * localMetres.y +
                    cameraForward(camera) * localMetres.z;
}

void cameraDolly(Camera& camera, float amount) {
  camera.position = camera.position + cameraForward(camera) * amount;
}

void cameraPan(Camera& camera, float dPixelX, float dPixelY, float viewportHeightPx) {
  if (viewportHeightPx <= 0.0F) {
    return;
  }
  float worldPerPixel = 2.0F * std::tan(camera.fovY * 0.5F) / viewportHeightPx; // at 1 m
  camera.position = camera.position - cameraRight(camera) * (dPixelX * worldPerPixel) +
                    cameraUp(camera) * (dPixelY * worldPerPixel);
}

} // namespace vredit
