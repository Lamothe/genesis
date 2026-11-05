// Regression for the shell's raw OpenXR view conversions; no XR runtime needed.
#include <cmath>
#include <cstdio>

#include "../app/xr_view.h"

using namespace vredit;

namespace {
int failures = 0;
#define CHECK(expr) do { if (!(expr)) { \
  std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); ++failures; \
} } while (false)

bool near(float a, float b) { return std::fabs(a - b) < 1e-5F; }

struct Frame {
  Mat4 eyeProj[2];
  Vec3 eyePos[2];
  Quat eyeOrient[2];
  Vec3 headPosition;
  Quat headOrient;
};

void checkPose(XrPosef const& p, XrPosef const& expected) {
  CHECK(p.position.x == expected.position.x);
  CHECK(p.position.y == expected.position.y);
  CHECK(p.position.z == expected.position.z);
  CHECK(p.orientation.x == expected.orientation.x);
  CHECK(p.orientation.y == expected.orientation.y);
  CHECK(p.orientation.z == expected.orientation.z);
  CHECK(p.orientation.w == expected.orientation.w);
}

void checkFov(XrFovf const& f, XrFovf const& expected) {
  CHECK(f.angleLeft == expected.angleLeft);
  CHECK(f.angleRight == expected.angleRight);
  CHECK(f.angleUp == expected.angleUp);
  CHECK(f.angleDown == expected.angleDown);
}
} // namespace

int main() {
  XrView views[2]{};
  views[0].pose = {{0.5F, 0.5F, 0.5F, 0.5F}, {-0.04F, 1.6F, 0.2F}};
  views[1].pose = {{0.5F, -0.5F, 0.5F, 0.5F}, {0.05F, 1.7F, 0.3F}};
  views[0].fov = {-0.9F, 0.7F, 0.8F, -0.6F};
  views[1].fov = {-0.6F, 1.0F, 0.7F, -0.9F};
  Frame frame{};
  inputFromXrViews(views, frame);

  for (int eye = 0; eye < 2; ++eye) {
    auto const& pose = views[eye].pose;
    CHECK(frame.eyePos[eye].x == pose.position.x);
    CHECK(frame.eyePos[eye].y == pose.position.y);
    CHECK(frame.eyePos[eye].z == pose.position.z);
    CHECK(frame.eyeOrient[eye].x == pose.orientation.x);
    CHECK(frame.eyeOrient[eye].y == pose.orientation.y);
    CHECK(frame.eyeOrient[eye].z == pose.orientation.z);
    CHECK(frame.eyeOrient[eye].w == pose.orientation.w);

    XrCompositionLayerProjectionView projection{};
    projection.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
    projection.subImage.imageRect.extent = {123, 456};
    projectionFromXrView(views[eye], projection);
    checkPose(projection.pose, views[eye].pose);
    checkFov(projection.fov, views[eye].fov);
    CHECK(projection.type == XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW);
    CHECK(projection.subImage.imageRect.extent.width == 123);
    CHECK(projection.subImage.imageRect.extent.height == 456);

    // At z=-1, each OpenXR tangent must map to the corresponding Vulkan NDC edge.
    auto const& m = frame.eyeProj[eye].m;
    auto const& f = views[eye].fov;
    CHECK(near(m[0] * std::tan(f.angleLeft) - m[8], -1.0F));
    CHECK(near(m[0] * std::tan(f.angleRight) - m[8], 1.0F));
    CHECK(near(m[5] * std::tan(f.angleUp) - m[9], -1.0F));
    CHECK(near(m[5] * std::tan(f.angleDown) - m[9], 1.0F));
    CHECK(m[11] == -1.0F);
    CHECK(near(m[10], 200.0F / (0.05F - 200.0F)));
    CHECK(near(m[14], (0.05F * 200.0F) / (0.05F - 200.0F)));
  }
  CHECK(frame.headPosition.x == (views[0].pose.position.x + views[1].pose.position.x) * 0.5F);
  CHECK(frame.headPosition.y == (views[0].pose.position.y + views[1].pose.position.y) * 0.5F);
  CHECK(frame.headPosition.z == (views[0].pose.position.z + views[1].pose.position.z) * 0.5F);
  CHECK(frame.headOrient.x == views[0].pose.orientation.x);
  CHECK(frame.headOrient.y == views[0].pose.orientation.y);
  CHECK(frame.headOrient.z == views[0].pose.orientation.z);
  CHECK(frame.headOrient.w == views[0].pose.orientation.w);

  Ray aim{{3.0F, 4.0F, 5.0F}, {0.0F, 1.0F, 0.0F}};
  Ray gaze = xrPointerRay(false, aim, views[0]);
  CHECK(gaze.origin.x == views[0].pose.position.x);
  CHECK(gaze.origin.y == views[0].pose.position.y);
  CHECK(gaze.origin.z == views[0].pose.position.z);
  CHECK(near(gaze.direction.x, -1.0F));
  CHECK(near(gaze.direction.y, 0.0F));
  CHECK(near(gaze.direction.z, 0.0F));
  Ray rightGaze = xrPointerRay(false, aim, views[1]);
  CHECK(rightGaze.origin.x == views[1].pose.position.x);
  CHECK(rightGaze.origin.y == views[1].pose.position.y);
  CHECK(rightGaze.origin.z == views[1].pose.position.z);
  CHECK(near(rightGaze.direction.x, 0.0F));
  CHECK(near(rightGaze.direction.y, 1.0F));
  CHECK(near(rightGaze.direction.z, 0.0F));
  Ray selected = xrPointerRay(true, aim, views[0]);
  CHECK(selected.origin.x == aim.origin.x && selected.origin.y == aim.origin.y && selected.origin.z == aim.origin.z);
  CHECK(selected.direction.x == aim.direction.x && selected.direction.y == aim.direction.y && selected.direction.z == aim.direction.z);

  std::printf("XR view conversion: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
