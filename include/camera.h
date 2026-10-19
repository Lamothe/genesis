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

// Vulkan clip-space projection for an asymmetric frustum given as tangent
// box (the tan() of OpenXR XrFovf angles; tanLeft/tanDown are typically
// negative). Matches cameraProjection's depth [0,1] and clip-Y-down
// conventions; the symmetric case is exactly cameraProjection at aspect 1.
Mat4 cameraProjectionFromTangents(float tanLeft, float tanRight, float tanUp, float tanDown,
                                  float nearZ, float farZ);

// World -> eye view matrix for a rigid pose (position + orientation), the
// inverse of the eye's world transform. Matches cameraView's conventions
// (view forward is -Z).
Mat4 cameraViewFromPose(Vec3 const& position, Quat const& orientation);

// Horizon lock: strip the roll from a tracked head orientation, keeping the
// look direction exactly where the user points it. Everyone carries a few
// degrees of physical head roll; composed with turning or a stick pitch it
// makes the horizon swing ("the viewer is tilted... not level with the ground
// as they turn"). The correction is a pure rotation about the head's forward
// axis: forward is preserved exactly and only the roll goes.
Quat gravityAlign(Quat orientation);

// Ray through a pixel; direction is normalised and points into the scene.
Ray cameraRay(Camera const& camera, float aspect, Vec2 pixel, Vec2 viewportSize);

// Mouse look: changes orientation only (pitch clamped short of gimbal flip).
void cameraLook(Camera& camera, float dYaw, float dPitch);

// Turn around a pivot in the camera's local frame instead of around the
// camera position: the given local point keeps its world position fixed while
// the orientation changes. VR smooth turn passes the tracked head here —
// pivoting on the tracking origin instead would swing the viewer in an arc
// around the origin. localPoint = {0, 0, 0} degenerates to cameraLook.
void cameraLookPivot(Camera& camera, Vec3 localPoint, float dYaw, float dPitch);
// Flies by a local offset in metres: x = right, y = up, z = forward along the
// current view direction (true fly movement, pitch included).
void cameraFly(Camera& camera, Vec3 localMetres);

// Free-flight movement for a head-tracked rig (VR): forward/back follows the
// head's LOOK direction with its pitch — looking down + forward descends,
// which is what adjusts your own height — while strafe stays level. The
// delta is in world metres; localMetres uses cameraFly's convention (x =
// right, y = up, z = forward-positive metres).
Vec3 cameraFlyLook(Camera const& camera, Vec3 headForwardLocal, Vec3 localMetres);

// Wheel dolly: > 0 flies forward along the view direction.
void cameraDolly(Camera& camera, float amount);

// Fine positioning pan in pixels: strafes right/up at a 1 m view scale.
void cameraPan(Camera& camera, float dPixelX, float dPixelY, float viewportHeightPx);

} // namespace vredit
