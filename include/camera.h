#pragma once

#include "vecmath.h"

namespace vredit {

// First-person free-flight camera: position + yaw/pitch orientation, +Y up and
// yaw 0 looking toward -Z (first-person VR-friendly: you fly through the scene
// instead of orbiting a pivot). Projection is right-handed with depth in
// [0, 1] (Vulkan/D3D convention, which is also what the neural seam expects)
// and Y flipped in clip space so world +Y is up on screen and pixel Y grows
// downward — matching screen-space pick maths.
struct Camera {
  Vec3 position{0.0F, 1.7F, 6.0F};
  float yaw{0.0F};     // radians around +Y, 0 = -Z
  float pitch{-0.15F}; // look elevation, clamped short of gimbal flip
  float fovY{1.0F};    // radians
  float nearZ{0.05F};
  float farZ{200.0F};
};

Vec3 cameraForward(Camera const& camera);
Vec3 cameraRight(Camera const& camera);
Vec3 cameraUp(Camera const& camera);
Mat4 cameraView(Camera const& camera);
Mat4 cameraProjection(Camera const& camera, float aspect);
Mat4 cameraViewProjection(Camera const& camera, float aspect);

// Ray through a pixel; direction is normalised and points into the scene.
Ray cameraRay(Camera const& camera, float aspect, Vec2 pixel, Vec2 viewportSize);

// Mouse look: changes orientation only (pitch clamped short of gimbal flip).
void cameraLook(Camera& camera, float dYaw, float dPitch);

// Flies by a local offset in metres: x = right, y = up, z = forward along the
// current view direction (true fly movement, pitch included).
void cameraFly(Camera& camera, Vec3 localMetres);

// Wheel dolly: > 0 flies forward along the view direction.
void cameraDolly(Camera& camera, float amount);

// Fine positioning pan in pixels: strafes right/up at a 1 m view scale.
void cameraPan(Camera& camera, float dPixelX, float dPixelY, float viewportHeightPx);

} // namespace vredit
