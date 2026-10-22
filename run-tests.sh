#!/usr/bin/env bash
# Thin launcher: compile and run the typed C++ core tests. All logic lives in C++.
set -euo pipefail
cd "$(dirname "$0")"
CXX_BIN="${CXX:-g++}"
mkdir -p .build

# Image decoder fixtures (generated assets, never checked in). Tests skip
# gracefully when these are missing.
FIX=.build/fixtures
mkdir -p "$FIX"
if command -v magick >/dev/null 2>&1; then
  magick -size 8x8 xc:'#ff0000' "$FIX/red8.png"
  magick -size 8x8 xc:'#204080' -depth 16 "$FIX/blue16.png"
  magick -size 8x8 xc:'gray(128)' "$FIX/grey8.png"
  magick -size 8x8 xc:'rgba(255,0,0,0.5)' "$FIX/alpha.png"
  magick -size 8x8 xc:red -colors 4 "PNG8:$FIX/palette.png"
  magick \( xc:'#ff0000' xc:'#0000ff' +append \) \( xc:'#0000ff' xc:'#ff0000' +append \) -append "$FIX/checker2.png"
  magick -size 2x2 xc:'rgb(128,255,128)' "$FIX/normal-up.png"  # flat tangent-space +Y normal
  magick -size 2x2 xc:'rgb(201,201,201)' "$FIX/normal-tri.png" # diagonal (+X+Y+Z) tangent normal
  magick -size 9000x9000 xc:'gray(80)' "$FIX/big9k.png"      # above the old 64M-pixel cap
  magick -size 20000x20000 xc:'gray(80)' "$FIX/huge20k.png"  # above any sane cap
  magick -size 8x8 xc:'#3060a0' -quality 95 "$FIX/blue.jpg"
  magick -size 8x8 xc:'gray(64)' -quality 95 "$FIX/grey.jpg"
  magick \( -size 8x8 xc:red xc:blue +append \) -quality 95 "$FIX/two.jpg"
  magick -size 8x8 xc:'#3060a0' -interlace plane -quality 95 "$FIX/progressive.jpg"
else
  echo "note: ImageMagick (magick) not found; image decoder tests will skip"
fi
"$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -O2 -Iinclude \
  src/scene.cpp src/picking.cpp src/gizmo.cpp src/camera.cpp src/handles.cpp \
  src/editor_state.cpp src/carousel.cpp \
  src/mesh.cpp src/json.cpp src/gltf.cpp src/obj.cpp src/image.cpp \
  tests/test_core.cpp \
  -lz -o .build/test-core
exec ./.build/test-core
