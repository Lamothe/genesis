#pragma once

#include "vecmath.h"

namespace vredit {

// Orbit camera around a target. Projection is right-handed with depth in
// [0, 1] (Vulkan/D3D convention, which is also what the neural seam expects)
// and Y flipped in clip space so world +Y is up on screen and pixel Y grows
// downward — matching screen-space pick maths.
struct Camera {
  Vec3 target{0.0F, 0.5F, 0.0F};
  float yaw{0.0F};     // radians around +Y
  float pitch{-0.35F}; // camera elevation, clamped short of gimbal flip
  float distance{6.0F};
  float fovY{1.0F}; // radians
  float nearZ{0.05F};
  float farZ{200.0F};
};

Vec3 cameraPosition(Camera const& camera);
Vec3 cameraForward(Camera const& camera);
Vec3 cameraRight(Camera const& camera);
Vec3 cameraUp(Camera const& camera);
Mat4 cameraView(Camera const& camera);
Mat4 cameraProjection(Camera const& camera, float aspect);
Mat4 cameraViewProjection(Camera const& camera, float aspect);

// Ray through a pixel; direction is normalised and points into the scene.
Ray cameraRay(Camera const& camera, float aspect, Vec2 pixel, Vec2 viewportSize);

void cameraOrbit(Camera& camera, float dYaw, float dPitch);
void cameraDolly(Camera& camera, float amount); // > 0 moves closer
void cameraPan(Camera& camera, float dPixelX, float dPixelY, float viewportHeightPx);

} // namespace vredit
