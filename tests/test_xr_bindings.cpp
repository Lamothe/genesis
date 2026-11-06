#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main() {
  std::ifstream input("app/xr_shell.cpp");
  std::string const source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  auto const touch = source.find("{\"/interaction_profiles/oculus/touch_controller\"");
  auto const vive = source.find("{\"/interaction_profiles/htc/vive_controller\"");
  auto const end = source.find("\n  };", vive);
  if (touch == std::string::npos || vive == std::string::npos || end == std::string::npos || touch >= vive) {
    std::cerr << "XR binding profiles missing\n";
    return 1;
  }
  auto const touchBindings = source.substr(touch, vive - touch);
  auto const viveBindings = source.substr(vive, end - vive);
  auto has = [](std::string const& text, char const* value) { return text.find(value) != std::string::npos; };
  if (!has(touchBindings, "/user/hand/right/input/b/click\", xr_->undoAction") ||
      !has(touchBindings, "/user/hand/left/input/y/click\", xr_->redoAction") ||
      !has(viveBindings, "/user/hand/left/input/trackpad\", xr_->leftStickAction") ||
      !has(viveBindings, "/user/hand/right/input/trackpad\", xr_->rightStickAction") ||
      !has(viveBindings, "/user/hand/left/input/trackpad/touch\", xr_->trackpadTouchAction") ||
      !has(viveBindings, "/user/hand/right/input/trackpad/touch\", xr_->trackpadTouchAction") ||
      has(touchBindings, "trackpadTouchAction") ||
      has(viveBindings, "trackpad/click") || has(viveBindings, "undoAction") || has(viveBindings, "redoAction")) {
    std::cerr << "XR history/navigation binding regression\n";
    return 1;
  }
  // The actual shell must wire history into all the same lifecycle points as
  // Menu/A, not just use the correct latch in an isolated unit test.
  auto count = [](std::string const& text, std::string const& needle) {
    std::size_t total = 0;
    for (std::size_t pos = 0; (pos = text.find(needle, pos)) != std::string::npos; pos += needle.size()) {
      ++total;
    }
    return total;
  };
  for (auto const* name : {"undo", "redo"}) {
    std::string const press = std::string("xr_->") + name + "Press";
    std::string const result = std::string(name) + "Result";
    std::string const state = std::string(name) + "State";
    if (count(source, press + ".interrupt();") != 4 ||
        !has(source, (press + ".updateTracked(toggleSafe, XR_SUCCEEDED(" + result + ") && " + state + ".isActive,").c_str()) ||
        !has(source, (std::string("input.") + name + "Pressed = " + press + ".pending();").c_str()) ||
        !has(source, (press + ".consume();").c_str())) {
      std::cerr << "XR history press lifecycle regression\n";
      return 1;
    }
  }
  // Keep the frame's real carousel toggle ahead of every trigger consumer.
  std::ifstream editorInput("app/editor_app.cpp");
  std::string const editor{std::istreambuf_iterator<char>(editorInput), std::istreambuf_iterator<char>()};
  auto const frame = editor.find("// Only the hand profile can drive gestures");
  auto const toggles = editor.find("processControllerUiToggles(app, renderer, input, headWorld, headForward)", frame);
  auto const panel = editor.find("app.panelPress.press(app.handHover);", frame);
  auto const viewer = editor.find("processControllerViewerPress(app, input,", frame);
  auto const placement = editor.find("app.carouselGate.update(true,", frame);
  if (frame == std::string::npos || toggles == std::string::npos || panel == std::string::npos ||
      viewer == std::string::npos || placement == std::string::npos ||
      toggles >= panel || toggles >= viewer || toggles >= placement) {
    std::cerr << "XR menu/carousel transition ordering regression\n";
    return 1;
  }
  std::cout << "XR binding regressions passed\n";
}
