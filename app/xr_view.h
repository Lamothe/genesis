#pragma once

#include <cmath>

#include <openxr/openxr.h>

#include "camera.h"

namespace vredit {

// The frame type supplies the eye and head fields; keeping this independent of
// the shell lets the conversion be tested without Vulkan or an OpenXR runtime.
template <typename Frame>
void inputFromXrViews(XrView const (&views)[2], Frame& input) {
  for (int eye = 0; eye < 2; ++eye) {
    XrFovf const& fov = views[eye].fov;
    input.eyeProj[eye] = cameraProjectionFromTangents(
        std::tan(fov.angleLeft), std::tan(fov.angleRight), std::tan(fov.angleUp),
        std::tan(fov.angleDown), 0.05F, 200.0F);
    XrPosef const& pose = views[eye].pose;
    input.eyePos[eye] = {pose.position.x, pose.position.y, pose.position.z};
    input.eyeOrient[eye] = {pose.orientation.x, pose.orientation.y, pose.orientation.z,
                            pose.orientation.w};
  }
  input.headPosition = (input.eyePos[0] + input.eyePos[1]) * 0.5F;
  input.headOrient = input.eyeOrient[0];
}

inline void projectionFromXrView(XrView const& view, XrCompositionLayerProjectionView& projection) {
  projection.pose = view.pose;
  projection.fov = view.fov;
}

inline Ray rayFromXrPose(XrPosef const& pose) {
  // OpenXR orientation is world <- local; forward is -Z, right is +X.
  Quat q{pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
  Vec3 forward = rotate(q, {0.0F, 0.0F, -1.0F});
  Vec3 origin{pose.position.x, pose.position.y, pose.position.z};
  return {origin, normalized(forward)};
}

inline Ray xrPointerRay(bool rightAimValid, Ray const& rightAim, XrView const& leftView) {
  return rightAimValid ? rightAim : rayFromXrPose(leftView.pose);
}

} // namespace vredit
