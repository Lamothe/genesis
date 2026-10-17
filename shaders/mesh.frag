#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColour;
layout(location = 2) in vec2 vUv;

layout(set = 0, binding = 1) uniform sampler2D uTextures[32];

layout(location = 0) out vec4 outColour;

layout(push_constant) uniform Push {
  mat4 model;
  vec4 tint;
  vec4 params; // x: lit, y: texture index, z: alpha cutoff, w: alpha mode
} push;

void main() {
  int textureIndex = int(push.params.y + 0.5);
  vec4 texel = texture(uTextures[textureIndex], vUv);
  vec3 n = normalize(vNormal);
  float ndl = abs(dot(n, normalize(vec3(0.35, 0.85, 0.45))));
  float light = mix(1.0, 0.35 + 0.65 * ndl, push.params.x);
  vec4 base = vec4(vColour * light, push.tint.a) * texel;
  float mode = push.params.w;
  if (mode > 0.5 && mode < 1.5 && base.a < push.params.z) {
    discard; // AlphaMode::Mask
  }
  outColour = base;
}
