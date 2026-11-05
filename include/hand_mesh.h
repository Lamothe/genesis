#pragma once

#include "mesh.h"

#include <cstddef>

namespace vredit {

constexpr std::size_t kHandJointCount = 26;

// Build one hand in the coordinate space of the input positions (metres), with
// OpenXR EXT_hand_tracking joint ordering. Works for either left or right hand;
// no handedness flag or joint orientation is needed. The source arrays must
// each contain 26 entries. Colour is linear RGB and UVs are not meaningful.
//
// The palm requires valid, finite wrist and index/middle/little proximal joints.
// Each digit requires all its own joints, including its tip. Invalid or
// implausibly collapsed/long parts are omitted rather than inferred. An empty
// result therefore denotes insufficient usable tracking, not a cached pose.
// This is an approximate procedural surface, not a user rig or anatomical scan;
// intersecting digit/palm surfaces, sharp bends and varying runtime joint
// conventions can leave seams. Regenerate on each tracked frame.
MeshData makeHandMesh(Vec3 const (&positions)[kHandJointCount],
                      bool const (&valid)[kHandJointCount],
                      Vec3 colour = {0.72F, 0.48F, 0.36F});

} // namespace vredit
