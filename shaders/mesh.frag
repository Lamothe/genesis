#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColour;

layout(location = 0) out vec4 outColour;

layout(push_constant) uniform Push {
  mat4 model;
  vec4 tint;
  vec4 params; // x: lit (1 = shaded, 0 = unlit), yzw reserved
} push;

void main() {
  vec3 n = normalize(vNormal);
  float ndl = abs(dot(n, normalize(vec3(0.35, 0.85, 0.45))));
  float light = mix(1.0, 0.35 + 0.65 * ndl, push.params.x);
  outColour = vec4(vColour * light, push.tint.a);
}
