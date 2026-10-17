#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColour;
layout(location = 3) in vec2 inUv;

layout(set = 0, binding = 0) uniform Frame {
  mat4 viewProj;
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
  vNormal = mat3(push.model) * inNormal;
  vColour = inColour * push.tint.rgb;
  vUv = inUv;
  vWorldPos = world.xyz;
}
