#!/usr/bin/env bash
# Thin launcher: compile and run the typed C++ core tests. All logic lives in C++.
set -euo pipefail
cd "$(dirname "$0")"
CXX_BIN="${CXX:-g++}"
mkdir -p .build
"$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude \
  src/scene.cpp src/picking.cpp src/gizmo.cpp src/camera.cpp src/handles.cpp \
  tests/test_core.cpp \
  -o .build/test-core
exec ./.build/test-core
