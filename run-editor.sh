#!/usr/bin/env bash
# Thin launcher: compile shaders and the typed C++ editor shell, then run it.
# All logic lives in C++ (project rule: no Python).
set -euo pipefail
cd "$(dirname "$0")"
CXX_BIN="${CXX:-g++}"
mkdir -p .build/shaders
glslc shaders/mesh.vert -o .build/shaders/mesh.vert.spv
glslc shaders/mesh.frag -o .build/shaders/mesh.frag.spv
"$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude \
  src/scene.cpp src/picking.cpp src/gizmo.cpp src/camera.cpp src/handles.cpp \
  src/mesh.cpp src/json.cpp src/gltf.cpp src/obj.cpp \
  app/geometry.cpp app/vk_renderer.cpp app/editor_app.cpp \
  -lvulkan -lxcb -o .build/vr-editor
exec ./.build/vr-editor "$@"
