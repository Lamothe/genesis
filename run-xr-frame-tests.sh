#!/usr/bin/env bash
# Shell lifecycle tests: no headset or active runtime required, but link the XR and Vulkan loaders.
set -euo pipefail
cd "$(dirname "$0")"
CXX_BIN="${CXX:-g++}"
mkdir -p .build
"$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude -Iapp -Ithird_party/openxr/include \
  tests/test_xr_frame.cpp src/camera.cpp -lvulkan -l:libopenxr_loader.so.1 -o .build/test-xr-frame
exec ./.build/test-xr-frame
