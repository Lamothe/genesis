# VR virtual pitch: unresolved rotation conflict

Baseline: main `697a2878109ecbd2c0bc4c4f43f83f025014c16e`.
This is an investigation, **not a fix**. Do not integrate it as a resolution.

The current shell applies one rigid tracking-to-world transform to both eyes,
controller poses and rays, and hand poses. With both sticks neutral, that
transform remains fixed; world objects and world-anchored menus therefore stay
fixed relative to the physical tracking space. The compositor receives the raw
located eye poses. Left-stick strafe follows the complete view-right axis,
including vertical motion when the head is rolled; a horizontal-only strafe
would change the requested controls.

## Why a fixed rig cannot satisfy all of the requested behaviour

Use +Y world-up, column vectors, and let X(a) and Y(a) denote rotations about
those axes. Start with a physically unrolled head X(-0.2). Eight full-deflection
50 ms virtual-pitch steps produce a rig rotation X(0.6). The physical head then
yaws to Y(1.1)X(-0.2), while the stick is neutral. The right vector becomes
`X(0.6)Y(1.1)(1,0,0)` and its vertical component is
`sin(0.6) sin(1.1)`, approximately 0.503. The resulting apparent horizon tilt
is not physical head roll. `tests/test_vr_locomotion.cpp` tests this actual
input sequence and its analytic value.

In general, if the same fixed rigid rig R is used for all tracked poses,
requiring a level right vector both before and after nonzero, non-180-degree physical
yaw implies the tracking-space world-up vector `transpose(R) * (0,1,0)` is
parallel to physical up. Such a rig cannot provide arbitrary virtual pitch.
Physical roll tracking does not remove this incompatibility.

## Choices requiring an explicit decision

1. **Retain the existing fixed rig and stick pitch.** The physical world,
   controller/hand relationships and menus stay spatially coherent, and stereo
   remains rigid, but the horizon can tilt disorientatingly after physical
   yaw. This does not satisfy the requested no-spurious-tilt requirement.
2. **Use head-dependent pitch/roll compensation.** One local example maps raw
   orientation `Y(yaw)X(pitch)Z(roll)` to
   `Y(yaw + virtualYaw)X(pitch + virtualPitch)Z(roll)` and constructs a
   shared rig from the result. It preserves instantaneous rigid stereo and
   controller alignment and the local roll parameter away from Euler
   singularities, but as physical yaw changes the rig must rotate even with
   sticks neutral. A physically stationary controller or room feature away
   from the head moves relative to the virtual scene; virtual world coordinates
   staying constant do not prevent that visible swimming. Euler pitch also has
   singularities near vertical, needing an explicit policy. This is a change in
   behaviour, not a free correction. World-anchored menus similarly move relative
   to the physical room. The test demonstrates the controller displacement.
3. **Keep the rig gravity-aligned and omit or reset virtual pitch.** Physical
   yaw and roll remain natural and the scene is stationary, but requested
   right-stick pitch is removed or changed. It must not be silently selected.

Non-rigid per-device or per-eye corrections instead sacrifice spatial/stereo
alignment; they are not an acceptable hidden workaround. Another interaction
model would need a clearly approved compromise, pole and tracking-loss policies,
and testing on an actual headset before any safety or comfort claim.
