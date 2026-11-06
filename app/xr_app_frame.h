#pragma once

#include "xr_shell.h"

namespace vredit {

// The editor shares these decisions with the headset-free app regression tests.
inline bool xrAppShouldStop(XrShell::FrameStatus status, bool& failed) {
  if (status == XrShell::FrameStatus::Failed) {
    failed = true;
    return true;
  }
  return status == XrShell::FrameStatus::Exit;
}

inline int xrAppExitCode(bool failed, bool smoke, int framesRendered) {
  return failed ? 2 : (smoke && framesRendered < 60 ? 1 : 0);
}

template <typename Shell, typename Renderer>
bool xrAppAbortOnFatalArmUpdate(Shell& shell, Renderer const& renderer) {
  if (!renderer.hasFatalError()) { return false; }
  shell.abortFrame();
  return true;
}

template <typename Shell, typename DrawEye>
bool xrAppDrawAndFinish(Shell& shell, DrawEye&& drawEye) {
  bool drew = true;
  for (int eye = 0; eye < 2; ++eye) {
    if (!drawEye(eye)) {
      drew = false;
      break;
    }
  }
  bool const ended = drew ? shell.endFrame() : shell.abortFrame();
  return drew && ended;
}

} // namespace vredit
