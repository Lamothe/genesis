#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUv;

layout(set = 0, binding = 0) uniform Frame {
  mat4 viewProj;
  vec4 cameraPos;
} frame;

layout(push_constant) uniform Push {
  mat4 model;
  vec4 tint;
  vec4 params; // x: lit, y: texture index, z: alpha cutoff, w: alpha mode
  vec4 normalParams; // x: has normal map, y: texture index, z: scale, w: highlight
} push;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColour;
layout(location = 2) out vec2 vUv;
layout(location = 3) out vec3 vWorldPos;

void main() {
  vec4 world = push.model * vec4(inPos, 1.0);
  gl_Position = frame.viewProj * world;
  mat3 linear = mat3(push.model);
  // Scaling first avoids a determinant threshold that depends on model size.
  float scale = max(max(max(abs(linear[0].x), abs(linear[0].y)), abs(linear[0].z)),
                    max(max(max(abs(linear[1].x), abs(linear[1].y)), abs(linear[1].z)),
                        max(max(abs(linear[2].x), abs(linear[2].y)), abs(linear[2].z))));
  vec3 normal = inNormal;
  if (scale > 0.0) {
    mat3 scaled = linear / scale;
    // A singular (or nearly singular) transform has no stable inverse.
    normal = abs(determinant(scaled)) > 1e-8
        ? transpose(inverse(scaled)) * inNormal : scaled * inNormal;
  }
  // Keep even zero normals and collapsed axes finite for fragment shading.
  vNormal = dot(normal, normal) > 1e-20 ? normal : vec3(0.0, 0.0, 1.0);
  vColour = inColour * push.tint.rgb;
  vUv = inUv;
  vWorldPos = world.xyz;
}
