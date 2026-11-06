// Exercise the actual shell frame lifecycle with a deterministic OpenXR stub.
#include "../app/xr_shell.cpp"
#include "../app/xr_app_frame.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

namespace {
std::vector<std::string> calls;
int failAcquire = -1;
int failWait = -1;
int waitFailures = 0;
int timeouts = 0;
int failRelease = -1;
int endLayers = -1;
bool endFails = false;
bool locateFails = false;
XrSessionState eventState = XR_SESSION_STATE_UNKNOWN;
bool sessionBeginFails = false;
bool sessionEndFails = false;
bool frameWaitFails = false;
bool frameBeginFails = false;
XrSwapchain chain(int eye) { return reinterpret_cast<XrSwapchain>(static_cast<std::uintptr_t>(eye + 1)); }
int eyeOf(XrSwapchain swapchain) { return static_cast<int>(reinterpret_cast<std::uintptr_t>(swapchain)) - 1; }
}

extern "C" {
XRAPI_ATTR XrResult XRAPI_CALL xrPollEvent(XrInstance, XrEventDataBuffer* event) {
  if (eventState == XR_SESSION_STATE_UNKNOWN) { return XR_EVENT_UNAVAILABLE; }
  auto* changed = reinterpret_cast<XrEventDataSessionStateChanged*>(event);
  changed->type = XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED;
  changed->state = eventState;
  eventState = XR_SESSION_STATE_UNKNOWN;
  return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrBeginSession(XrSession, XrSessionBeginInfo const*) {
  return sessionBeginFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrEndSession(XrSession) {
  return sessionEndFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrWaitFrame(XrSession, XrFrameWaitInfo const*, XrFrameState* state) {
  calls.emplace_back("waitFrame"); if (frameWaitFails) { return XR_ERROR_RUNTIME_FAILURE; } state->predictedDisplayTime = 123; state->shouldRender = XR_TRUE; return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrBeginFrame(XrSession, XrFrameBeginInfo const*) {
  calls.emplace_back("begin"); return frameBeginFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrLocateViews(XrSession, XrViewLocateInfo const*, XrViewState* state,
                                              uint32_t, uint32_t* count, XrView* views) {
  if (locateFails) { return XR_ERROR_RUNTIME_FAILURE; }
  state->viewStateFlags = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT |
                          XR_VIEW_STATE_POSITION_TRACKED_BIT | XR_VIEW_STATE_ORIENTATION_TRACKED_BIT;
  *count = 2;
  for (int eye = 0; eye < 2; ++eye) {
    views[eye].pose.orientation.w = 1.0F;
    views[eye].fov = {-0.7F, 0.7F, 0.7F, -0.7F};
  }
  return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrSyncActions(XrSession, XrActionsSyncInfo const*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrStringToPath(XrInstance, char const*, XrPath* path) { *path = 1; return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrGetCurrentInteractionProfile(XrSession, XrPath, XrInteractionProfileState*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateFloat(XrSession, XrActionStateGetInfo const*, XrActionStateFloat*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateBoolean(XrSession, XrActionStateGetInfo const*, XrActionStateBoolean*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStateVector2f(XrSession, XrActionStateGetInfo const*, XrActionStateVector2f*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrGetActionStatePose(XrSession, XrActionStateGetInfo const*, XrActionStatePose*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrLocateSpace(XrSpace, XrSpace, XrTime, XrSpaceLocation*) { return XR_SUCCESS; }
XRAPI_ATTR XrResult XRAPI_CALL xrAcquireSwapchainImage(XrSwapchain swapchain, XrSwapchainImageAcquireInfo const*, uint32_t* index) {
  auto const eye = eyeOf(swapchain); calls.push_back("acquire" + std::to_string(eye));
  *index = 0; return eye == failAcquire ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrWaitSwapchainImage(XrSwapchain swapchain, XrSwapchainImageWaitInfo const*) {
  auto const eye = eyeOf(swapchain); calls.push_back("wait" + std::to_string(eye));
  if (eye == failWait && timeouts > 0) { --timeouts; return XR_TIMEOUT_EXPIRED; }
  if (eye == failWait && waitFailures > 0) { --waitFailures; return XR_ERROR_RUNTIME_FAILURE; }
  return XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrReleaseSwapchainImage(XrSwapchain swapchain, XrSwapchainImageReleaseInfo const*) {
  auto const eye = eyeOf(swapchain); calls.push_back("release" + std::to_string(eye));
  return eye == failRelease ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
XRAPI_ATTR XrResult XRAPI_CALL xrEndFrame(XrSession, XrFrameEndInfo const* info) {
  calls.emplace_back("end"); endLayers = static_cast<int>(info->layerCount);
  assert(info->displayTime == 123);
  if (info->layerCount == 0) { assert(info->layers == nullptr); }
  else {
    assert(info->layerCount == 1 && info->layers != nullptr);
    auto const* layer = reinterpret_cast<XrCompositionLayerProjection const*>(info->layers[0]);
    assert(layer->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION && layer->viewCount == 2);
    assert(layer->views[0].subImage.swapchain == chain(0) && layer->views[1].subImage.swapchain == chain(1));
  }
  return endFails ? XR_ERROR_RUNTIME_FAILURE : XR_SUCCESS;
}
}

namespace vredit {
class XrShellFrameTest {
public:
  static void setup(XrShell& shell) {
    shell.xr_ = new XrShell::ImplData();
    shell.imagesAcquired_[0] = shell.imagesAcquired_[1] = false;
    shell.imagesWaited_[0] = shell.imagesWaited_[1] = false;
    shell.xr_->sessionRunning = true;
    shell.xr_->profileCheckIn = 100;
    for (int eye = 0; eye < 2; ++eye) {
      shell.xr_->swapchains[eye] = chain(eye);
      shell.xr_->images[eye].resize(1);
      shell.xr_->imageViews[eye].resize(1);
    }
  }
  static void teardown(XrShell& shell) {
    delete shell.xr_; shell.xr_ = nullptr;
  }
};
}

namespace {
struct MockShell {
  int aborts = 0;
  int ends = 0;
  bool finishOk = true;
  bool abortFrame() { ++aborts; return finishOk; }
  bool endFrame() { ++ends; return finishOk; }
};
struct MockRenderer {
  bool fatal = false;
  int failEye = -1;
  std::vector<int> eyes;
  bool hasFatalError() const { return fatal; }
  bool drawEye(int eye) { eyes.push_back(eye); return eye != failEye; }
};

// Mirrors the editor's decision points, using the same helpers it calls.
bool appFrame(MockShell& shell, MockRenderer& renderer) {
  if (vredit::xrAppAbortOnFatalArmUpdate(shell, renderer)) { return false; }
  return vredit::xrAppDrawAndFinish(shell, [&](int eye) { return renderer.drawEye(eye); });
}

void testAppDecisions() {
  using namespace vredit;
  bool failed = false;
  assert(xrAppShouldStop(XrShell::FrameStatus::Exit, failed));
  assert(!failed && xrAppExitCode(failed, false, 0) == 0);
  assert(xrAppShouldStop(XrShell::FrameStatus::Failed, failed));
  assert(failed && xrAppExitCode(failed, false, 60) == 2);
  assert(xrAppExitCode(false, true, 0) == 1);
  failed = false;
  assert(!xrAppShouldStop(XrShell::FrameStatus::Render, failed) && !failed);
  assert(!xrAppShouldStop(XrShell::FrameStatus::Skipped, failed) && !failed);
  assert(!xrAppShouldStop(XrShell::FrameStatus::Pump, failed) && !failed);
  MockShell armShell;
  MockRenderer armRenderer;
  armRenderer.fatal = true;
  assert(!appFrame(armShell, armRenderer));
  assert(armShell.aborts == 1 && armShell.ends == 0 && armRenderer.eyes.empty());
  for (int failEye = 0; failEye < 2; ++failEye) {
    MockShell shell;
    MockRenderer renderer;
    renderer.failEye = failEye;
    assert(!appFrame(shell, renderer));
    assert(shell.aborts == 1 && shell.ends == 0);
    assert(renderer.eyes.size() == static_cast<std::size_t>(failEye + 1));
    assert(renderer.eyes.back() == failEye);
  }
  MockShell shell;
  MockRenderer renderer;
  assert(appFrame(shell, renderer));
  assert(shell.ends == 1 && shell.aborts == 0 && (renderer.eyes == std::vector<int>{0, 1}));
  shell.finishOk = false;
  assert(!appFrame(shell, renderer));
  assert(shell.ends == 2 && shell.aborts == 0);
}
}

// Exercise app decisions through a real shell. Draws share the OpenXR call log
// so that draw, release and end ordering are checked together.
void testRealShellAppFrames() {
  using vredit::XrShell;
  enum class Scenario { FatalArm, FirstEyeFailure, SecondEyeFailure, Success, EndFailure, ReleaseFailure };
  for (auto const scenario : {Scenario::FatalArm, Scenario::FirstEyeFailure,
                              Scenario::SecondEyeFailure, Scenario::Success,
                              Scenario::EndFailure, Scenario::ReleaseFailure}) {
    XrShell shell;
    vredit::XrShellFrameTest::setup(shell);
    vredit::XrFrameInput input;
    MockRenderer renderer;
    renderer.fatal = scenario == Scenario::FatalArm;
    renderer.failEye = scenario == Scenario::FirstEyeFailure ? 0 :
                       scenario == Scenario::SecondEyeFailure ? 1 : -1;
    calls.clear();
    endLayers = -1;
    endFails = scenario == Scenario::EndFailure;
    failRelease = scenario == Scenario::ReleaseFailure ? 0 : -1;
    assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
    bool const aborted = vredit::xrAppAbortOnFatalArmUpdate(shell, renderer);
    bool const ok = !aborted && vredit::xrAppDrawAndFinish(shell, [&](int eye) {
      calls.push_back("draw" + std::to_string(eye));
      return renderer.drawEye(eye);
    });
    assert(aborted == (scenario == Scenario::FatalArm));
    assert(ok == (scenario == Scenario::Success));

    std::vector<std::string> expected = {"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1"};
    if (scenario != Scenario::FatalArm) { expected.emplace_back("draw0"); }
    if (scenario != Scenario::FatalArm && scenario != Scenario::FirstEyeFailure) {
      expected.emplace_back("draw1");
    }
    expected.insert(expected.end(), {"release0", "release1", "end"});
    assert(calls == expected);
    bool const projection = scenario == Scenario::Success || scenario == Scenario::EndFailure;
    assert(endLayers == (projection ? 1 : 0));
    assert(shell.frameStats().begun == 1 && shell.frameStats().zeroLayer == (projection ? 0U : 1U));
    assert(shell.frameStats().acquireFail == 0 && shell.frameStats().viewFail == 0 &&
           shell.frameStats().actionFail == 0);
    auto const size = calls.size();
    assert(!shell.endFrame() && !shell.abortFrame() && calls.size() == size);
    vredit::XrShellFrameTest::teardown(shell);
  }
  endFails = false;
  failRelease = -1;
  calls.clear();
}

int main() {
  testAppDecisions();
  testRealShellAppFrames();
  using vredit::XrShell;
  XrShell shell;
  assert(!shell.endFrame() && !shell.abortFrame());
  vredit::XrShellFrameTest::setup(shell);
  vredit::XrFrameInput input;
  assert(!shell.abortFrame() && !shell.endFrame() && calls.empty());
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed); // no nested begin
  assert(shell.abortFrame());
  assert(endLayers == 0 && shell.frameStats().begun == 1 && shell.frameStats().zeroLayer == 1);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "release0", "release1", "end"}));
  auto size = calls.size();
  assert(!shell.abortFrame() && !shell.endFrame() && calls.size() == size);

  calls.clear();
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
  assert(shell.endFrame() && endLayers == 1);
  assert(shell.frameStats().begun == 2 && shell.frameStats().zeroLayer == 1);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "release0", "release1", "end"}));

  calls.clear(); failAcquire = 1;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Skipped && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "release0", "end"}));
  assert(shell.frameStats().acquireFail == 1 && shell.frameStats().zeroLayer == 2);
  assert(!shell.endFrame());

  calls.clear(); failAcquire = -1; failWait = 1; waitFailures = 1;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Skipped && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "release0", "wait1", "release1", "end"}));
  assert(shell.frameStats().acquireFail == 2 && shell.frameStats().zeroLayer == 3);

  calls.clear(); failWait = 1; timeouts = 1;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
  assert(shell.endFrame() && endLayers == 1);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "wait1", "release0", "release1", "end"}));

  calls.clear(); failWait = 1; waitFailures = 2;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "release0", "wait1", "end"}));
  assert(!shell.endFrame());
  // Simulate teardown after an unrecoverable runtime wait failure.
  vredit::XrShellFrameTest::teardown(shell);
  vredit::XrShellFrameTest::setup(shell);

  calls.clear(); failWait = -1; failRelease = 0;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
  assert(!shell.endFrame() && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "wait1", "release0", "release1", "end"}));
  assert(shell.frameStats().zeroLayer == 5);
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  vredit::XrShellFrameTest::teardown(shell);
  vredit::XrShellFrameTest::setup(shell);

  calls.clear(); failRelease = -1; failAcquire = 1; endFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "release0", "end"}));
  assert(!shell.endFrame());

  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  vredit::XrShellFrameTest::teardown(shell);
  vredit::XrShellFrameTest::setup(shell);
  failAcquire = -1; endFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Render);
  assert(!shell.abortFrame() && !shell.endFrame());
  assert(shell.frameStats().begun == 9 && shell.frameStats().zeroLayer == 7);
  vredit::XrShellFrameTest::teardown(shell);
  vredit::XrShellFrameTest::setup(shell);
  calls.clear(); endFails = false; locateFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Skipped && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "end"}));
  assert(shell.frameStats().begun == 10 && shell.frameStats().zeroLayer == 8 && shell.frameStats().viewFail == 1);
  vredit::XrShellFrameTest::teardown(shell);
  vredit::XrShellFrameTest::setup(shell);
  locateFails = false;
  eventState = XR_SESSION_STATE_EXITING;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Exit);
  eventState = XR_SESSION_STATE_READY; sessionBeginFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  sessionBeginFails = false;
  eventState = XR_SESSION_STATE_STOPPING; sessionEndFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  sessionEndFails = false;
  eventState = XR_SESSION_STATE_READY;
  frameWaitFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  frameWaitFails = false; frameBeginFails = true;
  assert(shell.beginFrame(input) == XrShell::FrameStatus::Failed);
  frameBeginFails = false;
  calls.clear(); failAcquire = 1; failRelease = 0;
  auto const cleanupStatus = shell.beginFrame(input);
  assert(cleanupStatus == XrShell::FrameStatus::Failed && endLayers == 0);
  assert((calls == std::vector<std::string>{"waitFrame", "begin", "acquire0", "wait0", "acquire1", "release0", "end"}));
  bool appFailed = false;
  assert(vredit::xrAppShouldStop(cleanupStatus, appFailed));
  assert(vredit::xrAppExitCode(appFailed, false, 0) == 2);
  vredit::XrShellFrameTest::teardown(shell);
  std::puts("XR frame lifecycle tests passed");
}
