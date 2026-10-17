#pragma once

#include <cmath>

namespace vredit {

struct Vec3 {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
};

struct Vec2 {
  float x{0.0F};
  float y{0.0F};
};

struct Ray {
  Vec3 origin{};
  Vec3 direction{}; // expected normalised
};

struct Aabb {
  Vec3 min{};
  Vec3 max{};
};

inline constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline constexpr Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
inline constexpr Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline constexpr Vec3 operator*(float s, Vec3 a) { return a * s; }
inline constexpr Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline constexpr Vec3 operator/(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }

inline constexpr float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline constexpr Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }

inline Vec3 normalized(Vec3 a) {
  float len = length(a);
  if (len <= 0.0F) {
    return {};
  }
  return a / len;
}

struct Quat {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  float w{1.0F};
};

inline constexpr Quat operator*(Quat a, Quat b) {
  // Hamilton product: (a * b) applies b first, then a.
  return {
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

inline constexpr Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }

inline Quat normalized(Quat q) {
  float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n <= 0.0F) {
    return {};
  }
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}

inline Quat quatFromAxisAngle(Vec3 axis, float radians) {
  Vec3 n = normalized(axis);
  float half = radians * 0.5F;
  float s = std::sin(half);
  return {n.x * s, n.y * s, n.z * s, std::cos(half)};
}

inline constexpr Vec3 rotate(Quat q, Vec3 v) {
  Vec3 u{q.x, q.y, q.z};
  Vec3 t = cross(u, v) * 2.0F;
  return v + t * q.w + cross(u, t);
}

struct Mat4 {
  // Column-major: m[column * 4 + row].
  float m[16] = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
};

inline Mat4 mul(Mat4 const& a, Mat4 const& b) {
  Mat4 r;
  for (int c = 0; c < 4; ++c) {
    for (int row = 0; row < 4; ++row) {
      r.m[c * 4 + row] = a.m[row] * b.m[c * 4 + 0] + a.m[4 + row] * b.m[c * 4 + 1] +
                         a.m[8 + row] * b.m[c * 4 + 2] + a.m[12 + row] * b.m[c * 4 + 3];
    }
  }
  return r;
}

inline Mat4 translationMatrix(Vec3 t) {
  Mat4 r;
  r.m[12] = t.x;
  r.m[13] = t.y;
  r.m[14] = t.z;
  return r;
}

inline Mat4 scaleMatrix(Vec3 s) {
  Mat4 r;
  r.m[0] = s.x;
  r.m[5] = s.y;
  r.m[10] = s.z;
  return r;
}

inline Mat4 rotationMatrix(Quat q) {
  float xx = q.x * q.x;
  float yy = q.y * q.y;
  float zz = q.z * q.z;
  float xy = q.x * q.y;
  float xz = q.x * q.z;
  float yz = q.y * q.z;
  float wx = q.w * q.x;
  float wy = q.w * q.y;
  float wz = q.w * q.z;
  Mat4 r;
  r.m[0] = 1.0F - 2.0F * (yy + zz);
  r.m[1] = 2.0F * (xy + wz);
  r.m[2] = 2.0F * (xz - wy);
  r.m[4] = 2.0F * (xy - wz);
  r.m[5] = 1.0F - 2.0F * (xx + zz);
  r.m[6] = 2.0F * (yz + wx);
  r.m[8] = 2.0F * (xz + wy);
  r.m[9] = 2.0F * (yz - wx);
  r.m[10] = 1.0F - 2.0F * (xx + yy);
  return r;
}

inline Mat4 trs(Vec3 t, Quat r, Vec3 s) {
  return mul(translationMatrix(t), mul(rotationMatrix(r), scaleMatrix(s)));
}

inline Vec3 transformPoint(Mat4 const& a, Vec3 p) {
  return {
      a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12],
      a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13],
      a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14],
  };
}

inline Vec3 transformVector(Mat4 const& a, Vec3 v) {
  return {
      a.m[0] * v.x + a.m[4] * v.y + a.m[8] * v.z,
      a.m[1] * v.x + a.m[5] * v.y + a.m[9] * v.z,
      a.m[2] * v.x + a.m[6] * v.y + a.m[10] * v.z,
  };
}

// Exact bounds of the transformed box: transforms all 8 corners.
inline Aabb transformAabb(Mat4 const& a, Aabb const& box) {
  Aabb out{};
  for (int corner = 0; corner < 8; ++corner) {
    Vec3 local{(corner & 1) != 0 ? box.max.x : box.min.x,
               (corner & 2) != 0 ? box.max.y : box.min.y,
               (corner & 4) != 0 ? box.max.z : box.min.z};
    Vec3 point = transformPoint(a, local);
    if (corner == 0) {
      out = {point, point};
    } else {
      out.min = {std::fmin(out.min.x, point.x), std::fmin(out.min.y, point.y),
                 std::fmin(out.min.z, point.z)};
      out.max = {std::fmax(out.max.x, point.x), std::fmax(out.max.y, point.y),
                 std::fmax(out.max.z, point.z)};
    }
  }
  return out;
}

inline Aabb unite(Aabb const& a, Aabb const& b) {
  return {{std::fmin(a.min.x, b.min.x), std::fmin(a.min.y, b.min.y),
           std::fmin(a.min.z, b.min.z)},
          {std::fmax(a.max.x, b.max.x), std::fmax(a.max.y, b.max.y),
           std::fmax(a.max.z, b.max.z)}};
}

inline Vec3 aabbCentre(Aabb const& box) {
  return (box.min + box.max) * 0.5F;
}

// Inverts the affine part (last row 0,0,0,1). Returns false if singular.
inline bool invertAffine(Mat4 const& a, Mat4& out) {
  Vec3 c0{a.m[0], a.m[1], a.m[2]};
  Vec3 c1{a.m[4], a.m[5], a.m[6]};
  Vec3 c2{a.m[8], a.m[9], a.m[10]};
  float det = dot(c0, cross(c1, c2));
  if (std::fabs(det) < 1e-12F) {
    return false;
  }
  float inv = 1.0F / det;
  Vec3 r0 = cross(c1, c2) * inv;
  Vec3 r1 = cross(c2, c0) * inv;
  Vec3 r2 = cross(c0, c1) * inv;
  Vec3 t{a.m[12], a.m[13], a.m[14]};
  out = Mat4{};
  out.m[0] = r0.x;
  out.m[1] = r1.x;
  out.m[2] = r2.x;
  out.m[4] = r0.y;
  out.m[5] = r1.y;
  out.m[6] = r2.y;
  out.m[8] = r0.z;
  out.m[9] = r1.z;
  out.m[10] = r2.z;
  out.m[12] = -dot(r0, t);
  out.m[13] = -dot(r1, t);
  out.m[14] = -dot(r2, t);
  return true;
}

// Decomposes an affine matrix into translation/rotation/scale. Assumes a
// shear-free matrix (TRS hierarchies; nested non-uniform scale under rotation can
// introduce shear, which TRS cannot represent). Mirrors are representable: a
// negative determinant becomes a negative X scale by convention.
inline void decompose(Mat4 const& a, Vec3& t, Quat& r, Vec3& s) {
  t = {a.m[12], a.m[13], a.m[14]};
  Vec3 c0{a.m[0], a.m[1], a.m[2]};
  Vec3 c1{a.m[4], a.m[5], a.m[6]};
  Vec3 c2{a.m[8], a.m[9], a.m[10]};
  float det = dot(c0, cross(c1, c2));
  s = {length(c0), length(c1), length(c2)};
  if (det < 0.0F) {
    s.x = -s.x;
  }
  if (s.x != 0.0F) {
    c0 = c0 / s.x;
  }
  if (s.y != 0.0F) {
    c1 = c1 / s.y;
  }
  if (s.z != 0.0F) {
    c2 = c2 / s.z;
  }
  float m00 = c0.x;
  float m01 = c1.x;
  float m02 = c2.x;
  float m10 = c0.y;
  float m11 = c1.y;
  float m12 = c2.y;
  float m20 = c0.z;
  float m21 = c1.z;
  float m22 = c2.z;
  float trace = m00 + m11 + m22;
  if (trace > 0.0F) {
    float s2 = std::sqrt(trace + 1.0F) * 2.0F;
    r = Quat{(m21 - m12) / s2, (m02 - m20) / s2, (m10 - m01) / s2, 0.25F * s2};
  } else if (m00 > m11 && m00 > m22) {
    float s2 = std::sqrt(1.0F + m00 - m11 - m22) * 2.0F;
    r = Quat{0.25F * s2, (m01 + m10) / s2, (m02 + m20) / s2, (m21 - m12) / s2};
  } else if (m11 > m22) {
    float s2 = std::sqrt(1.0F + m11 - m00 - m22) * 2.0F;
    r = Quat{(m01 + m10) / s2, 0.25F * s2, (m12 + m21) / s2, (m02 - m20) / s2};
  } else {
    float s2 = std::sqrt(1.0F + m22 - m00 - m11) * 2.0F;
    r = Quat{(m02 + m20) / s2, (m12 + m21) / s2, 0.25F * s2, (m10 - m01) / s2};
  }
  r = normalized(r);
}

} // namespace vredit
