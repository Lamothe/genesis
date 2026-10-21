#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColour;
layout(location = 2) in vec2 vUv;
layout(location = 3) in vec3 vWorldPos;

layout(set = 0, binding = 1) uniform sampler2D uTextures[256];

layout(location = 0) out vec4 outColour;

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

void main() {
  int textureIndex = int(push.params.y + 0.5);
  vec4 texel = texture(uTextures[textureIndex], vUv);
  vec3 n = normalize(vNormal);
  if (push.normalParams.x > 0.5) {
    // Tangent-space normal map with a derivative-based TBN. Solving the 2x2
    // system for dW/du and dW/dv directly (Cramer) is correct in any screen
    // handedness and reproduces glTF's (tangent=+U, bitangent=+V) frame,
    // including mirrored UVs, with no sign juggling.
    vec3 mapN = texture(uTextures[int(push.normalParams.y + 0.5)], vUv).xyz * 2.0 - 1.0;
    mapN.xy *= push.normalParams.z;
    vec3 dp1 = dFdx(vWorldPos);
    vec3 dp2 = dFdy(vWorldPos);
    vec2 duv1 = dFdx(vUv);
    vec2 duv2 = dFdy(vUv);
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) > 1e-12) {
      vec3 t = (dp1 * duv2.y - dp2 * duv1.y) / det;
      vec3 b = (dp2 * duv1.x - dp1 * duv2.x) / det;
      t = t - n * dot(n, t);
      b = b - n * dot(n, b);
      float tLen = length(t);
      float bLen = length(b);
      if (tLen > 1e-6 && bLen > 1e-6) {
        n = normalize((t / tLen) * mapN.x + (b / bLen) * mapN.y + n * mapN.z);
      }
    }
  }
  float ndl = abs(dot(n, normalize(vec3(0.35, 0.85, 0.45))));
  float light = mix(1.0, 0.35 + 0.65 * ndl, push.params.x);
  vec4 base = vec4(vColour * light, push.tint.a) * texel;
  float mode = push.params.w;
  if (mode > 0.5 && mode < 1.5 && base.a < push.params.z) {
    discard; // AlphaMode::Mask
  }
  // glTF: OPAQUE ignores texture alpha and kept MASK fragments are opaque;
  // only BLEND keeps alpha. Without this the layer's passthrough composite
  // would show the real world through the model wherever the texture has
  // an alpha channel (and MASK cutouts would render half transparent).
  if (mode < 1.5) {
    base.a = 1.0;
  }
  // State highlight: 5 = selected (a THIN warm rim on grazing-angle pixels
  // — the model's own shading is untouched; the user rejected fill tints as
  // too heavy), 3..4 = delete charge (red ramp: 3.0 means charge 0, 4.0
  // means "about to delete").
  float highlight = push.normalParams.w;
  if (highlight > 4.5) {
    vec3 toCamera = normalize(frame.cameraPos.xyz - vWorldPos);
    float rim = pow(1.0 - abs(dot(n, toCamera)), 3.0);
    base.rgb += vec3(0.95, 0.62, 0.25) * rim * 0.55;
  } else if (highlight > 2.5) {
    float charge = clamp(highlight - 3.0, 0.0, 1.0);
    base.rgb = mix(base.rgb, vec3(1.0, 0.12, 0.08), 0.30 + 0.45 * charge);
  }
  outColour = base;
}
