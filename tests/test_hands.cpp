#include "controller_workflow.h"
#include "hand_input.h"

#include <cstdio>
#include <initializer_list>
#include <limits>

using namespace vredit;

namespace {
int checks = 0;
int failures = 0;

void check(bool condition, char const* expression, int line) {
  ++checks;
  if (!condition) {
    ++failures;
    std::printf("FAIL test_hands.cpp:%d: %s\n", line, expression);
  }
}
#define CHECK(...) check((__VA_ARGS__), #__VA_ARGS__, __LINE__)

bool near(Vec3 a, Vec3 b) {
  return length(a - b) < 2e-5F;
}

bool near(Mat4 const& a, Mat4 const& b) {
  for (int i = 0; i < 16; ++i) {
    if (!(std::fabs(a.m[i] - b.m[i]) < 2e-5F)) {
      return false;
    }
  }
  return true;
}


void leftTriggerWorkflow() {
  using A = LeftTriggerWorkflow::Action;
  LeftTriggerWorkflow w;
  auto sample = [&](bool enabled, bool active, bool held, bool pressed, bool released,
                    bool right, bool grabbing, bool poses, float dt = .1F) {
    return w.update(enabled, active, held, pressed, released, right, grabbing, poses, dt);
  };
  CHECK(sample(true, true, true, true, false, false, false, true) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::none);
  CHECK(sample(true, true, true, true, false, false, false, true) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::clearSelection);
  CHECK(sample(true, true, true, true, false, false, false, true) == A::none);
  CHECK(sample(true, true, true, false, false, false, false, true, .4F) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::clearSelection);
  CHECK(sample(true, true, true, true, false, true, true, true) == A::beginScale);
  CHECK(w.scaling());
  CHECK(sample(true, true, true, false, false, true, true, true) == A::none);
  CHECK(sample(true, true, false, false, true, true, true, true) == A::endScale);
  CHECK(!w.scaling());
  CHECK(sample(true, true, true, true, false, true, true, false) == A::none);
  CHECK(sample(true, true, false, false, true, true, true, true) == A::none);
  CHECK(sample(true, true, true, true, false, true, true, true) == A::beginScale);
  CHECK(sample(true, true, true, false, false, true, true, false) == A::trackingLost);
  CHECK(sample(true, true, true, false, false, true, true, true) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::none);
  CHECK(sample(true, true, true, true, false, false, false, true) == A::none);
  CHECK(sample(false, true, true, false, false, false, false, true) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::none);
  CHECK(sample(true, true, true, true, false, false, false, true) == A::none);
  CHECK(sample(true, true, true, false, false, true, true, true) == A::none);
  CHECK(sample(true, true, false, false, true, true, true, true) == A::none);
  CHECK(sample(true, true, true, true, false, true, true, true) == A::beginScale);
  CHECK(sample(true, true, true, false, false, false, false, true) == A::none);
  CHECK(sample(true, true, false, false, true, false, false, true) == A::none);
  CHECK(sample(true, true, true, true, false, true, true, true) == A::beginScale);
  CHECK(sample(true, false, true, false, false, true, true, true) == A::trackingLost);
}


void booleanRearmLatch() {
  BooleanRearmLatch latch;
  // No initial hold can start until an active release has been observed.
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(false, false));
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, false));
  CHECK(latch.update(true, true));
  CHECK(latch.update(true, true));
  CHECK(!latch.update(true, false));
  CHECK(latch.update(true, true));

  // A synthetic inactive release cannot rearm a hold, even over many frames.
  CHECK(!latch.update(false, false));
  CHECK(!latch.update(false, true));
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, false)); // real release silently rearms
  CHECK(latch.update(true, true));

  // Explicit interruption covers skipped frames, focus and profile changes;
  // even an already released input needs an active release after interruption.
  latch.interrupt();
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, false));
  CHECK(latch.update(true, true));
  CHECK(!latch.update(true, false));
  latch.interrupt();
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, false));
  CHECK(latch.update(true, true));

  // View loss while charging cancels and blocks until a release on a tracked
  // frame. An apparent release during view loss must not rearm it.
  CHECK(!latch.updateTracked(false, true, false));
  CHECK(!latch.updateTracked(true, true, true));
  CHECK(!latch.updateTracked(true, false, false));
  CHECK(!latch.updateTracked(true, true, true));
  CHECK(!latch.updateTracked(true, true, false));
  CHECK(latch.updateTracked(true, true, true));

  // An interrupt after a hold has fired also blocks repeat until release.
  latch.interrupt();
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(false, false));
  CHECK(!latch.update(true, true));
  CHECK(!latch.update(true, false));
  CHECK(latch.update(true, true));
}


void togglePressInterruptions() {
  // Menu and A use identical edge semantics. No input query failure, inactive
  // action, focus/profile interruption, skipped frame or lost view is a release.
  for (int button = 0; button < 2; ++button) {
    BooleanPressLatch latch;
    latch.updateTracked(true, true, true);
    CHECK(!latch.pending());
    latch.updateTracked(true, true, false);
    latch.updateTracked(true, true, true);
    CHECK(latch.pending());
    latch.consume();
    latch.updateTracked(true, true, true);
    CHECK(!latch.pending());

    // A held button that becomes inactive cannot manufacture a second edge.
    latch.updateTracked(true, false, false);
    latch.updateTracked(true, true, true);
    CHECK(!latch.pending());
    latch.updateTracked(true, false, false);
    latch.updateTracked(true, true, true);
    CHECK(!latch.pending());
    latch.updateTracked(true, true, false);
    latch.updateTracked(true, true, true);
    CHECK(latch.pending());

    // A pending press is discarded on loss, not replayed on reacquisition.
    latch.updateTracked(false, true, false);
    CHECK(!latch.pending());
    latch.updateTracked(false, true, true);
    CHECK(!latch.pending());
    latch.updateTracked(true, true, true);
    CHECK(!latch.pending());
    latch.updateTracked(true, true, false);
    CHECK(!latch.pending());
    latch.updateTracked(true, true, true);
    CHECK(latch.pending());

    // Explicit interruption covers focus/profile changes and skipped frames,
    // including a press sampled just before image acquisition fails.
    for (int reason = 0; reason < 3; ++reason) {
      latch.interrupt();
      CHECK(!latch.pending());
      latch.updateTracked(true, true, true);
      CHECK(!latch.pending());
      latch.updateTracked(true, true, false);
      latch.updateTracked(true, true, true);
      CHECK(latch.pending());
    }
    latch.consume();
    latch.updateTracked(true, true, false);
    latch.updateTracked(true, true, true);
    CHECK(latch.pending());
  }
}

void deleteContextInterruptions() {
  // Selection, carousel and menu changes all use the same rearm rule.
  for (int transition = 0; transition < 5; ++transition) {
    BooleanRearmLatch x;
    ToolPanelPress panel;
    CHECK(!x.updateTracked(true, true, false));
    CHECK(x.updateTracked(true, true, true));
    panel.press(ToolPanelPress::kDelete);
    CHECK(panel.hold(true, ToolPanelPress::kDelete, .3F) == ToolPanelPress::Action::none);
    CHECK(panel.charge() > 0.0F);
    x.interrupt();
    panel.cancel();
    CHECK(panel.charge() == 0.0F);
    CHECK(panel.hold(true, ToolPanelPress::kDelete, 1.0F) == ToolPanelPress::Action::none);
    CHECK(!x.updateTracked(true, true, true));
    CHECK(!x.updateTracked(true, false, false));
    CHECK(!x.updateTracked(false, true, false));
    CHECK(!x.updateTracked(true, true, true));
    CHECK(!x.updateTracked(true, true, false));
    CHECK(x.updateTracked(true, true, true));
    panel.press(ToolPanelPress::kDelete);
    CHECK(panel.hold(true, ToolPanelPress::kDelete, .6F) == ToolPanelPress::Action::deleteNow);
  }
}

void historyPressInterruptions() {
  // Exercise B and Y independently with the same sample/consume lifecycle as
  // the shell, including interruptions both before and after a sampled edge.
  for (int button = 0; button < 2; ++button) {
    BooleanPressLatch latch;
    auto frame = [&](bool safe, bool active, bool held) {
      latch.updateTracked(safe, active, held);
      bool const pressed = latch.pending();
      latch.consume();
      return pressed;
    };
    CHECK(!frame(true, true, false));
    CHECK(frame(true, true, true));
    CHECK(!frame(true, true, true)); // real uninterrupted hold is one edge
    CHECK(!frame(true, true, false));
    for (int reason = 0; reason < 3; ++reason) {
      // Skipped frame, focus loss or profile change with a previously-up button.
      latch.interrupt();
      CHECK(!frame(true, true, true));
      CHECK(!frame(true, true, true));
      CHECK(!frame(true, true, false));
      CHECK(frame(true, true, true));
      CHECK(!frame(true, true, false));
    }
    // A query failure/inactive action and an untracked view cannot rearm.
    CHECK(!frame(true, false, false));
    CHECK(!frame(true, true, true));
    CHECK(!frame(false, true, false));
    CHECK(!frame(true, true, true));
    CHECK(!frame(true, true, false));
    CHECK(frame(true, true, true));
    // A press sampled on a frame whose image acquisition fails is discarded.
    CHECK(!frame(true, true, false));
    latch.updateTracked(true, true, true);
    CHECK(latch.pending());
    latch.interrupt();
    CHECK(!latch.pending());
    CHECK(!frame(true, true, true));
    CHECK(!frame(true, true, false));
    CHECK(frame(true, true, true));
  }
}

void selectTriggerLatch() {
  SelectTriggerLatch latch;
  auto step = [&](bool active, float value, bool expectedActive, bool held,
                  bool pressed, bool released) {
    auto const output = latch.update(active, value);
    CHECK(output.active == expectedActive);
    CHECK(output.held == held);
    CHECK(output.pressed == pressed);
    CHECK(output.released == released);
  };

  // Initial acquisition and inactive samples cannot be mistaken for release.
  step(false, 0.0F, false, true, false, false);
  step(true, 1.0F, true, true, false, false);
  step(true, 0.41F, true, true, false, false);
  step(true, 0.4F, true, false, false, false); // silent rearm
  step(true, 0.55F, true, false, false, false);
  step(true, 0.551F, true, true, true, false);
  step(true, 0.55F, true, true, false, false);
  step(true, 0.401F, true, true, false, false);
  step(true, 0.4F, true, false, false, true);
  step(true, 0.4F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);

  // An inactive sample during a hold ends neither as a release nor a press;
  // held reacquisition stays blocked until a real active release.
  step(false, 0.0F, false, true, false, false);
  step(true, 1.0F, true, true, false, false);
  step(true, 0.4F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);
  step(true, std::numeric_limits<float>::quiet_NaN(), false, true, false, false);
  step(true, 1.0F, true, true, false, false);
  step(true, 0.0F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);

  // Focus/profile events and skipped frames call the same interruption hook.
  // Model the shell pending flag: even an active released sample on the event
  // frame is withheld until a rendered frame consumes that pending event.
  latch.interrupt(); // focus loss
  step(false, 0.0F, false, true, false, false); // pending event
  step(true, 1.0F, true, true, false, false);
  step(true, 0.0F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);
  latch.interrupt(); // profile change without an inactive action sample
  step(false, 0.0F, false, true, false, false); // pending event
  step(true, 1.0F, true, true, false, false);
  step(true, 0.0F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);
  latch.interrupt(); // skipped frame: no sample was delivered
  step(true, 1.0F, true, true, false, false);
  step(true, 0.0F, true, false, false, false);
  step(true, 0.6F, true, true, true, false);
  step(true, 0.0F, true, false, false, true);
}


void graspArbitration() {
  HandGraspGate gate;
  HandGesture grasp;
  auto step = [&](bool valid, float squeeze, bool pinch, bool captured, bool primed) {
    grasp.update(valid, squeeze);
    bool const held = handPinchHeld(valid, squeeze);
    gate.update(valid, held, pinch, captured);
    return gate.canAcquire(grasp.pressed(), primed);
  };
  CHECK(!step(true, 1, false, false, true)); // first sample cannot arm
  CHECK(!step(true, 0, false, false, true));
  CHECK(step(true, .8F, true, false, true)); // simultaneous: grasp wins
  CHECK(step(true, .8F, true, false, true)); // held may enter contact later
  CHECK(!step(false, 1, true, false, true));
  CHECK(!step(true, 1, true, false, true)); // reacquisition needs release
  CHECK(!step(true, 0, false, false, true));
  CHECK(!step(true, .8F, true, false, false)); // first pinch cannot move
  gate.reset(); grasp.reset(); // selection was made during this hold
  CHECK(!step(true, .8F, true, false, true));
  CHECK(!step(true, 0, true, false, true)); // both must release
  CHECK(!step(true, .8F, false, false, true));
  CHECK(!step(true, 0, false, false, true));
  CHECK(!step(true, .8F, false, false, false)); // unprimed selection remains safe
  CHECK(!step(true, .8F, true, true, true)); // UI capture owns overlap
  CHECK(!step(true, .8F, false, false, true));
  CHECK(!step(true, 0, false, false, true));
  CHECK(step(true, .8F, true, false, true));
}

void contactCacheLifetime() {
  HandContactCache cache;
  auto geometry = std::make_shared<PickGeometry>();
  geometry->positions = {{0,0,0},{1,0,0},{0,1,0}};
  geometry->indices = {0,1,2};
  std::weak_ptr<const PickGeometry> weak = geometry;
  auto mesh = cache.get(geometry);
  CHECK(cache.get(geometry) == mesh);
  cache.prune();
  CHECK(cache.size() == 1);
  geometry.reset();
  cache.prune(); // an external mesh still needs its geometry
  CHECK(cache.size() == 1 && !weak.expired());
  CHECK(mesh->near({.2F,.2F,0}, Mat4{}, .035F));
  mesh.reset();
  cache.prune();
  CHECK(cache.size() == 0 && weak.expired());
}

void surfaceContact() {
  auto g = std::make_shared<PickGeometry>();
  g->positions = {{0,0,0},{1,0,0},{0,1,0}, {100,100,100}};
  g->indices = {0,1,2, 0,0,0, 0,1,999}; // ignore invalid and degenerate triangles
  HandContactMesh mesh(g);
  CHECK(mesh.near({.2F,.2F,.03F}, Mat4{}, .035F));
  CHECK(!mesh.near({.8F,.8F,0}, Mat4{}, .035F)); // inside bounds, outside triangle
  CHECK(!mesh.near({.2F,.2F,.04F}, Mat4{}, .035F));
  CHECK(mesh.near({.51F,.51F,0}, Mat4{}, .035F)); // nearest edge
  CHECK(!mesh.near({100,100,100}, Mat4{}, .035F));
  auto const world = mul(translationMatrix({2,3,4}),
      mul(rotationMatrix(quatFromAxisAngle({0,1,0}, .6F)), scaleMatrix({4,.1F,2})));
  auto const surface = transformPoint(world, {.2F,.2F,0});
  auto const normal = rotate(quatFromAxisAngle({0,1,0}, .6F), {0,0,1});
  CHECK(mesh.near(surface+normal*.03F, world, .035F));
  CHECK(!mesh.near(surface+normal*.04F, world, .035F));
  CHECK(!mesh.near(surface, scaleMatrix({0,1,1}), .035F));
  CHECK(!mesh.near({NAN,0,0}, Mat4{}, .035F));
  CHECK(!mesh.near({0,0,0}, Mat4{}, NAN));
  CHECK(!mesh.near({0,0,0}, Mat4{}, -1));
  // Exercise internal BVH traversal and rejected distant regions.
  auto many = std::make_shared<PickGeometry>();
  for (unsigned i = 0; i < 1000; ++i) {
    float const x = static_cast<float>(i)*2;
    auto const base = static_cast<unsigned>(many->positions.size());
    many->positions.insert(many->positions.end(), {{x,0,0},{x+1,0,0},{x,1,0}});
    many->indices.insert(many->indices.end(), {base,base+1,base+2});
  }
  HandContactMesh large(many);
  CHECK(large.near({1998.2F,.2F,.02F}, Mat4{}, .035F));
  CHECK(!large.near({1999.2F,.2F,.02F}, Mat4{}, .035F));
  HandGesture squeeze;
  CHECK(!squeeze.update(true, 1)); // acquisition cannot start held
  CHECK(!squeeze.update(true, 0));
  CHECK(squeeze.update(true, .8F));
  CHECK(!squeeze.update(false, 1)); // readiness/tracking loss disarms
  CHECK(!squeeze.update(true, 1));
  CHECK(!squeeze.update(true, 0));
  CHECK(squeeze.update(true, .8F));
  HandPose const start{{1,2,3}, {}, true};
  auto const turn = quatFromAxisAngle({0,1,0}, 1.0F);
  HandPose const current{{1.1F,2.2F,3.3F}, turn, true};
  auto const object = translationMatrix({1.2F,2,3});
  auto const result = handGrabTransform(start, current, start.position, object);
  CHECK(near(transformPoint(result, {}), current.position + rotate(turn, {.2F,0,0})));
}

void gestures() {
  HandGesture gesture;
  CHECK(!gesture.update(false, 0.0F));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.5F));
  CHECK(!gesture.update(true, 0.4F));
  CHECK(!gesture.update(true, 0.549F));
  CHECK(gesture.update(true, 0.55F));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.41F));
  CHECK(!gesture.update(true, 0.55F));
  CHECK(!gesture.update(true, 0.4F));
  CHECK(gesture.update(true, 0.55F));
  CHECK(!gesture.update(false, 0.0F));
  CHECK(!gesture.update(true, 0.9F));
  CHECK(!gesture.update(true, 0.45F));
  CHECK(!gesture.update(true, 0.9F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 0.8F));
  gesture.reset();
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, std::numeric_limits<float>::quiet_NaN()));
  CHECK(!gesture.update(true, 1.0F));
  CHECK(!gesture.update(true, 0.0F));
  CHECK(gesture.update(true, 0.6F));
}

void selectionGraceAndTracking() {
  HandSelectionLatch latch;
  HandGesture gesture;
  CHECK(!gesture.update(true, 0));
  CHECK(gesture.update(true, 1));
  CHECK(!latch.startsGrab(true, true)); // first pinch cannot move anything
  latch.selected(true);
  CHECK(!latch.recentHit()); // selecting is not a post-release hover
  CHECK(!gesture.update(true, 1));
  CHECK(!gesture.update(true, 0));
  latch.rememberHit();
  latch.advance(.10F);
  CHECK(latch.recentHit());
  CHECK(gesture.update(true, 1));
  CHECK(latch.startsGrab(false, latch.recentHit()));
  latch.advance(.051F);
  CHECK(!latch.recentHit());
  CHECK(!latch.startsGrab(false, latch.recentHit()));
  CHECK(latch.startsGrab(true)); // exact hit still works after expiry
  latch.rememberHit();
  latch.clearHit(); // a different closest object, UI or stick capture
  CHECK(!latch.recentHit());
  latch.rememberHit();
  latch.selected(false); // selection changed/cleared
  CHECK(!latch.startsGrab(true, latch.recentHit()));
  latch.selected(true);
  latch.rememberHit();
  latch.advance(std::numeric_limits<float>::quiet_NaN());
  CHECK(!latch.recentHit());
  latch.rememberHit();
  latch.reset(); // tracking interruption requires new selection
  CHECK(!latch.recentHit());
  CHECK(!latch.startsGrab(true));

  // Aim loss during a held pinch must not reset the gesture. Pinch action or
  // palm tracking loss still stops edits and requires release to re-arm.
  HandGesture held;
  CHECK(!held.update(handPinchSignal(true, true, 0), 0));
  CHECK(held.update(handPinchSignal(true, true, .8F), .8F));
  bool const aimValid = false;
  CHECK(!aimValid && handPinchHeld(handPinchSignal(true, true, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, .8F), .8F));
  CHECK(held.pressed());
  CHECK(!handPinchHeld(handPinchSignal(true, false, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, false, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, .8F), .8F));
  CHECK(!held.update(handPinchSignal(true, true, 0), 0));
  CHECK(held.update(handPinchSignal(true, true, .8F), .8F));
  held.reset(); // palm tracking loss during grab
  CHECK(!held.update(true, .8F));
  CHECK(!held.update(true, 0));
  CHECK(held.update(true, .8F));
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(!handPinchHeld(handPinchSignal(true, true, nan), nan));
  CHECK(!handPinchSignal(false, true, .8F));
}

void carouselActions() {
  Vec3 const origin{0.0F, 0.0F, 0.0F};
  Vec3 const forward{0.0F, 0.0F, -1.0F};
  Vec3 const right{1.0F, 0.0F, 0.0F};
  Vec3 const centre{0.0F, 0.0F, -1.25F};
  auto const nan = std::numeric_limits<float>::quiet_NaN();

  // ANY side browses: the old +-0.22 central dead zone is gone. The crossing
  // point at ring depth decides, so the pointer origin never skews the zones.
  for (float const side : {-.5F, -.219F, -.001F, .001F, .219F, .5F}) {
    for (Vec3 const pointer : {origin, right * .7F + Vec3{0, -.3F, 0}}) {
      Ray const ray{pointer, normalized(centre + right * side - pointer)};
      CHECK(carouselPointerAction(false, ray, origin, forward) == (side < 0.0F ? -1 : 1));
    }
    // The exact hit on the settled centre preview places and has PRIORITY
    // over the side rules, wherever on the surface it lands.
    Ray const hitRay{origin, normalized(centre + right * side - origin)};
    CHECK(carouselPointerAction(true, hitRay, origin, forward) == 2);
  }

  // Ring-relative zones stay correct as the viewer turns and moves.
  for (float const yaw : {0.0F, 1.1F, -2.3F}) {
    auto const orientation = quatFromAxisAngle({0, 1, 0}, yaw);
    auto const fwd = rotate(orientation, {0, 0, -1});
    auto const rgt = rotate(orientation, {1, 0, 0});
    Vec3 const eye{2, 1, -3};
    Vec3 const ring = eye + fwd * 1.25F;
    for (Vec3 const pointer : {eye, eye + rgt * .7F + Vec3{0, -.3F, 0}}) {
      for (float const side : {-.5F, -.001F, .001F, .5F}) {
        Ray const ray{pointer, normalized(ring + rgt * side - pointer)};
        CHECK(carouselPointerAction(false, ray, eye, fwd) == (side < 0.0F ? -1 : 1));
        CHECK(carouselPointerAction(true, ray, eye, fwd) == 2);
      }
    }
  }

  // Exact-centre crossings (exact binary arithmetic) break the tie by where
  // the ray POINTS: crossing dead centre while aiming right browses next,
  // aiming left browses previous...
  Ray const centreAimingRight{{-0.5F, 0.0F, -0.25F}, {0.5F, 0.0F, -1.0F}};
  Ray const centreAimingLeft{{0.5F, 0.0F, -0.25F}, {-0.5F, 0.0F, -1.0F}};
  CHECK(carouselPointerAction(false, centreAimingRight, origin, forward) == 1);
  CHECK(carouselPointerAction(false, centreAimingLeft, origin, forward) == -1);
  // ...and exactly centred aim from a centred pointer is the single point
  // where left and right meet: inert (or place, on the exact hit).
  Ray const deadCentre{{0.0F, 0.0F, -0.25F}, {0.0F, 0.0F, -1.0F}};
  CHECK(carouselPointerAction(false, deadCentre, origin, forward) == 0);
  CHECK(carouselPointerAction(true, deadCentre, origin, forward) == 2);

  // A parallel ray never crosses the ring plane, so it browses by where it
  // points; a vertical one (pointing nowhere sideways) falls back to where
  // the pointer SITS relative to the ring centre.
  Ray const parallelRight{origin + forward * .25F, right};
  Ray const parallelLeft{origin + forward * .25F, -right};
  CHECK(carouselPointerAction(false, parallelRight, origin, forward) == 1);
  CHECK(carouselPointerAction(false, parallelLeft, origin, forward) == -1);
  Ray const verticalFromRight{right * .4F + forward * .25F, {0.0F, 1.0F, 0.0F}};
  Ray const verticalFromLeft{-right * .4F + forward * .25F, {0.0F, 1.0F, 0.0F}};
  CHECK(carouselPointerAction(false, verticalFromRight, origin, forward) == 1);
  CHECK(carouselPointerAction(false, verticalFromLeft, origin, forward) == -1);
  CHECK(carouselPointerAction(false, Ray{forward * .25F, {0.0F, 1.0F, 0.0F}},
                              origin, forward) == 0);

  // Backward rays and pointers that have walked past the ring plane are
  // handled by the same fallback, and a grazing surface clip from ANY of
  // them is never an inadvertent placement.
  Ray const backward{origin + forward * .25F, -forward};
  Ray const backwardFromRight{right * .4F + forward * .25F, -forward};
  CHECK(carouselPointerAction(false, backward, origin, forward) == 0);
  CHECK(carouselPointerAction(false, backwardFromRight, origin, forward) == 1);
  CHECK(carouselPointerAction(true, backwardFromRight, origin, forward) == 0);
  Ray const beyondRing{centre + forward * .75F, forward};
  Ray const beyondRingRight{right * .4F + centre + forward * .75F, forward};
  CHECK(carouselPointerAction(false, beyondRing, origin, forward) == 0);
  CHECK(carouselPointerAction(true, beyondRing, origin, forward) == 0);
  CHECK(carouselPointerAction(false, beyondRingRight, origin, forward) == 1);
  CHECK(carouselPointerAction(true, beyondRingRight, origin, forward) == 0);
  CHECK(carouselPointerAction(true, parallelRight, origin, forward) == 0);
  CHECK(carouselPointerAction(true, backward, origin, forward) == 0);
  // A zero-length direction is no aim at all.
  CHECK(carouselPointerAction(false, {origin + forward * .25F, {}}, origin, forward) == 0);
  CHECK(carouselPointerAction(true, {origin + forward * .25F, {}}, origin, forward) == 0);
  // A zero-length direction is inert even when the pointer is OFF-CENTRE:
  // the centred case above passes through the pointer-side tie-break.
  CHECK(carouselPointerAction(false, {right * .4F + forward * .25F, {}}, origin, forward) == 0);
  CHECK(carouselPointerAction(true, {right * .4F + forward * .25F, {}}, origin, forward) == 0);

  // The zone frame is LEVEL: a pitched ring forward is flattened to its
  // horizontal heading (the caller latches `ringForward` level already, but
  // the rule belongs to this function).
  Vec3 const pitched{0.0F, 0.7F, -0.7F};
  CHECK(carouselPointerAction(false, parallelRight, origin, pitched) == 1);
  CHECK(carouselPointerAction(false, parallelLeft, origin, pitched) == -1);
  CHECK(carouselPointerAction(true, {origin, pitched}, origin, pitched) == 2);

  // Near-parallel aims resolve the same way on both sides of the crossing
  // threshold: just below it the fallback says next, just above it the
  // crossing point (far away laterally) agrees.
  Ray const justBelow{{0.0F, 0.0F, -0.25F}, {1.0F, 0.0F, -5e-5F}};
  Ray const justAbove{{0.0F, 0.0F, -0.25F}, {1.0F, 0.0F, -1.5e-4F}};
  CHECK(carouselPointerAction(false, justBelow, origin, forward) == 1);
  CHECK(carouselPointerAction(false, justAbove, origin, forward) == 1);

  // Non-unit directions are rays too (the caller never normalises them).
  Ray const fastForward{origin + forward * .25F, forward * 2.0F};
  CHECK(carouselPointerAction(true, fastForward, origin, forward) == 2);
  Ray const fastRight{origin + forward * .25F, (right + forward) * 2.0F};
  CHECK(carouselPointerAction(false, fastRight, origin, forward) == 1);

  // Vertical offsets never change the horizontal zones.
  Ray const highRay{{0.0F, 1.3F, -0.25F}, {0.0F, 0.0F, -1.0F}};
  Ray const highSide{{0.0F, 1.3F, -0.25F}, normalized(right * .5F + forward)};
  CHECK(carouselPointerAction(true, highRay, origin, forward) == 2);
  CHECK(carouselPointerAction(false, highSide, origin, forward) == 1);

  // Non-finite and degenerate frames are inert — never a placement.
  CHECK(carouselPointerAction(false, {{}, {nan, 0, -1}}, origin, forward) == 0);
  CHECK(carouselPointerAction(false, {{nan, 0, 0}, {0, 0, -1}}, origin, forward) == 0);
  CHECK(carouselPointerAction(false, {{}, {0, 0, -1}}, {nan, 0, 0}, forward) == 0);
  CHECK(carouselPointerAction(false, {{}, {0, 0, -1}}, origin, {nan, 0, -1}) == 0);
  CHECK(carouselPointerAction(true, {{}, {nan, 0, -1}}, origin, forward) == 0);
  CHECK(carouselPointerAction(false, {{}, {0, 0, -1}}, origin, {}) == 0);
  CHECK(carouselPointerAction(true, {{}, {0, 0, -1}}, origin, {}) == 0);
  // A vertical "forward" leaves the ring frame undefined.
  CHECK(carouselPointerAction(false, {{}, {0, 0, -1}}, origin, {0, 1, 0}) == 0);
  CHECK(carouselPointerAction(true, {{}, {0, 0, -1}}, origin, {0, 1, 0}) == 0);
}

void panels() {
  HandPanel panel;
  auto front = [](Vec3 point) { return Ray{point + Vec3{0, 0, 1}, {0, 0, -1}}; };
  CHECK(panel.hit(front(handPanelButtonCentre(0))) == -1);
  panel.visible = true;
  for (int i : {0, 8, 9, 10, 11}) {
    auto const centre = handPanelButtonCentre(i);
    int const slot = i == 0 ? 0 : i - 7;
    CHECK(near(centre, {(slot % 2 == 0) ? -0.115F : 0.115F,
                        -0.065F * (static_cast<float>(slot / 2) + 0.5F), 0}));
    CHECK(panel.hit(front(centre)) == i);
    CHECK(panel.hit({centre - Vec3{0, 0, 1}, {0, 0, 1}}) == -1);
    CHECK(panel.hit({centre - Vec3{0, 0, 1}, {0, 0, -1}}) == -1);
    CHECK(panel.hit(front(centre + Vec3{0.1049F, 0.0279F, 0})) == i);
    CHECK(panel.hit(front(centre - Vec3{0.1049F, 0.0279F, 0})) == i);
    CHECK(panel.hit(front(centre + Vec3{0.1051F, 0, 0})) == -1);
    CHECK(panel.hit(front(centre + Vec3{0, 0.0281F, 0})) == -1);
  }
  // Exact inclusive edges on representable coordinates.
  auto const centre = handPanelButtonCentre(0);
  CHECK(panel.hit(front({centre.x + 0.105F, centre.y, 0})) == 0);
  CHECK(panel.hit(front({centre.x, centre.y + 0.028F, 0})) == 0);
  CHECK(panel.hit(front({0, -0.1F, 0})) == -1); // column gap
  CHECK(panel.hit(front({-0.115F, -0.065F, 0})) == -1); // row gap
  CHECK(panel.hit(front({0, 0, 0})) == -1);
  CHECK(panel.hit(front({0, -0.5F, 0})) == -1);
  CHECK(panel.hit({{0, 0, 1}, {1, 0, 0}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, -0.0000001F}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, 0}}) == -1);
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(panel.hit({{nan, 0, 1}, {0, 0, -1}}) == -1);
  CHECK(panel.hit({{0, 0, 1}, {0, 0, nan}}) == -1);
  CHECK(near(handPanelButtonCentre(-1), Vec3{}));
  CHECK(near(handPanelButtonCentre(12), Vec3{}));
  CHECK(near(handPanelButtonCentre(1000000), Vec3{}));

  panel.origin = {3, 2, -4};
  panel.orientation = quatFromAxisAngle({0, 1, 0}, 1.1F) *
                      quatFromAxisAngle({1, 0, 0}, 0.4F);
  for (int i : {0, 8, 9, 10, 11}) {
    auto const world = panel.origin + rotate(panel.orientation, handPanelButtonCentre(i));
    auto const normal = rotate(panel.orientation, {0, 0, 1});
    CHECK(panel.hit({world + normal, -normal}) == i);
    CHECK(panel.hit({world - normal, normal}) == -1);
  }
}

void palms() {
  HandPose pose;
  CHECK(!palmFacingUp(pose, false));
  pose.valid = true;
  CHECK(!palmFacingUp(pose, false)); // identity palm points down
  pose.orientation = quatFromAxisAngle({1, 0, 0}, 3.14159265F);
  CHECK(palmFacingUp(pose, false));
  pose.orientation = quatFromAxisAngle({1, 0, 0}, std::acos(-0.65F));
  CHECK(!palmFacingUp(pose, false));
  CHECK(palmFacingUp(pose, true));
  pose.orientation = quatFromAxisAngle({1, 0, 0}, std::acos(-0.5F));
  CHECK(!palmFacingUp(pose, true));
  pose.orientation = quatFromAxisAngle({0, 0, 1}, 3.14159265F);
  CHECK(palmFacingUp(pose, false));
  // Gesture is physical: virtual tracking-space pitch must not alter the
  // raw palm classification, even though its rendered world pose changes.
  HandPose const raw = pose;
  HandPose world = raw;
  world.orientation = normalized(quatFromAxisAngle({1, 0, 0}, 1.5F) * raw.orientation);
  CHECK(palmFacingUp(raw, false));
  CHECK(!palmFacingUp(world, false));
  pose.valid = false;
  CHECK(!palmFacingUp(pose, true));
}

void projection() {
  HandGrabView const view{{0,0,0}, {}, true};
  Vec3 const hand{.1F, .1F, -.5F};
  Vec3 const moved{.15F, .08F, -.5F};
  for (float depth : {1.0F, 10.0F, 100.0F}) {
    Vec3 const surface{depth*.2F, depth*.1F, -depth};
    Vec3 const target = projectedGrabPoint(view, hand, moved, surface);
    CHECK(std::fabs(target.x / depth - surface.x / depth - .1F) < .0001F);
    CHECK(std::fabs(target.y / depth - surface.y / depth + .04F) < .0001F);
    HandPose const start{hand, {}, true};
    HandPose const now{moved, quatFromAxisAngle({0, 1, 0}, .4F), true};
    auto const delta = projectedHandGrabTransform(view, start, now, surface, Mat4{});
    CHECK(length(transformPoint(delta, surface) - target) < .0001F);
  }
  // Depth is relative to the grab-start hand and surface depth: a 10%
  // push/pull moves near and distant surfaces by the same percentage.
  for (float depth : {0.3F, 1.0F, 10.0F, 100.0F}) {
    Vec3 const surface{depth * .2F, depth * .1F, -depth};
    auto const away = projectedGrabPoint(view, hand, {hand.x, hand.y, -.55F}, surface);
    auto const closer = projectedGrabPoint(view, hand, {hand.x, hand.y, -.45F}, surface);
    CHECK(std::fabs(-away.z / depth - 1.1F) < .0001F);
    CHECK(std::fabs(-closer.z / depth - .9F) < .0001F);
    // Depth-only motion preserves the existing projection calculation.
    CHECK(std::fabs(away.x / -away.z - (surface.x / depth + hand.x / .55F - hand.x / .5F)) < .0001F);
    CHECK(std::fabs(closer.y / -closer.z - (surface.y / depth + hand.y / .45F - hand.y / .5F)) < .0001F);
    CHECK(near(projectedGrabPoint(view, hand, hand, surface), surface));
  }
  // The projection clamp must not clamp depth, including a grab started
  // inside the projection near plane. Starting in place must never jump.
  auto const closeHand = projectedGrabPoint(view, hand, {.1F,.1F,-.05F}, {0,0,-10});
  CHECK(std::fabs(closeHand.z + 1.0F) < .0001F);
  Vec3 const nearStart{.01F, .02F, -.099F};
  Vec3 const farSurface{2,1,-10};
  CHECK(near(projectedGrabPoint(view, nearStart, nearStart, farSurface), farSurface));
  auto const nearPush = projectedGrabPoint(view, nearStart, {.01F,.02F,-.198F}, farSurface);
  CHECK(std::fabs(nearPush.z + 19.9F) < .0001F);
  auto const rayStart = Vec3{.06F, 0, -.06F};
  auto const sameRay = projectedGrabPoint(view, rayStart, {.1F,0,-.1F}, {0,0,-10});
  CHECK(std::fabs(sameRay.x) < .0001F);
  auto const tinyStart = projectedGrabPoint(view, {0,0,-.0011F}, {.1F,0,-.0111F}, {0,0,-10});
  auto const behindStart = projectedGrabPoint(view, {0,0,.05F}, {.1F,0,.05F}, {0,0,-10});
  CHECK(near(tinyStart, {.1F,0,-10.01F}));
  CHECK(near(behindStart, {.1F,0,-10}));
  auto const behind = projectedGrabPoint(view, hand, {.1F,.1F,.01F}, {0,0,-10});
  CHECK(std::fabs(behind.z + .05F) < .0001F);
  // Depth and lateral motion combine without losing the proportional gain.
  auto const combined = projectedGrabPoint(view, hand, {.15F, .08F, -.55F}, {2,1,-10});
  CHECK(std::fabs(combined.z + 11.0F) < .0001F);
  CHECK(std::fabs(combined.x / 11.0F - (.2F + .15F / .55F - .1F / .5F)) < .0001F);
  CHECK(std::fabs(combined.y / 11.0F - (.1F + .08F / .55F - .1F / .5F)) < .0001F);
  // When the hand crosses the near plane, the mapping remains bounded and continuous.
  auto const nearDepthA = projectedGrabPoint(view, hand, {.2F,.1F,-.10001F}, {0,0,-10});
  auto const nearDepthB = projectedGrabPoint(view, hand, {.2F,.1F,-.09999F}, {0,0,-10});
  CHECK(length(nearDepthA - nearDepthB) < .01F);
  CHECK(-nearDepthB.z >= .05F);
  auto const orientation = quatFromAxisAngle({0, 1, 0}, .7F);
  HandGrabView const turned{{2,1,3}, orientation, true};
  auto world = [&](Vec3 v) { return turned.eye + rotate(orientation, v); };
  auto const target = projectedGrabPoint(turned, world(hand), world(moved), world({2,1,-10}));
  CHECK(length(target - world({3,.6F,-10})) < .0001F);
  auto const nearA = projectedGrabPoint(view, hand, {.2F,.1F,-.10001F}, {0,0,-10});
  auto const nearB = projectedGrabPoint(view, hand, {.2F,.1F,-.09999F}, {0,0,-10});
  CHECK(length(nearA - nearB) < .01F);
  auto const fallback = projectedGrabPoint(view, {0,0,0}, {1,0,0}, {0,0,-10});

  CHECK(near(fallback, {1,0,-10}));
}

void grabs() {

  HandPose const start{{1, 2, 3},
                       quatFromAxisAngle({1, 0, 0}, 0.4F) *
                       quatFromAxisAngle({0, 1, 0}, -0.7F), true};
  Vec3 const pivot{2, -1, 0.5F};
  Mat4 const initial = trs({4, 2, -3}, quatFromAxisAngle({0, 0, 1}, 0.3F), {2, 3, 4});
  CHECK(near(handGrabTransform(start, start, pivot, initial), initial));
  auto slight = start;
  slight.orientation = quatFromAxisAngle({0, 1, 0}, .0002F) * start.orientation;
  auto const slightResult = handGrabTransform(start, slight, pivot, initial);
  CHECK(length(transformVector(slightResult, {1, 0, 0}) - transformVector(initial, {1, 0, 0})) > .0001F);

  for (Vec3 const axis : {Vec3{0, 1, 0}, Vec3{1, 0, 0}, Vec3{0, 0, 1}}) {
    auto const delta = quatFromAxisAngle(axis, 1.2F);
    HandPose current = start;
    current.orientation = delta * start.orientation;
    current.position = start.position + Vec3{-0.3F, 0.2F, 0.7F};
    auto const result = handGrabTransform(start, current, pivot, initial);
    // Independent point/vector expectations catch quaternion order, pivot,
    // translation order, and preservation of initial orientation/scale.
    for (Vec3 const point : {Vec3{}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}}) {
      CHECK(near(transformPoint(result, point),
                 pivot + (current.position - start.position) +
                 rotate(delta, transformPoint(initial, point) - pivot)));
    }
    CHECK(near(transformVector(result, {1, 2, 3}),
               rotate(delta, transformVector(initial, {1, 2, 3}))));
  }
  HandPose current = start;
  current.position = start.position + Vec3{0.25F, -0.5F, 1.5F};
  auto const expected = mul(translationMatrix((current.position - start.position)), initial);
  for (int i = 0; i < 10000; ++i) {
    CHECK(near(handGrabTransform(start, current, pivot, initial), expected));
  }
  // Returning to the captured start must undo every intermediate sample.
  for (int i = 0; i < 1000; ++i) {
    current.position = start.position + Vec3{static_cast<float>(i) * 0.001F, 0, 0};
    current.orientation = quatFromAxisAngle({0, 1, 0}, static_cast<float>(i) * 0.01F) *
                          start.orientation;
    (void)handGrabTransform(start, current, pivot, initial);
  }
  CHECK(near(handGrabTransform(start, start, pivot, initial), initial));
  current.valid = false;
  CHECK(near(handGrabTransform(start, current, pivot, initial), initial));
  CHECK(near(handGrabTransform(current, start, pivot, initial), initial));
}

void controllerPoseQuality() {
  ControllerPose pose;
  CHECK(!pose.usable());
  pose.valid = true;
  CHECK(!pose.usable()); // VALID alone is not measured tracking
  pose.tracked = true;
  CHECK(pose.usable());
  CHECK(pose.handPose().valid);
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  pose.position.x = nan;
  CHECK(!pose.usable());
  CHECK(!pose.handPose().valid);
  pose.position.x = 0;
  pose.orientation.w = nan;
  CHECK(!pose.usable());
  pose.orientation.w = 1;
  pose.valid = false;
  CHECK(!pose.usable()); // TRACKED without VALID is still unusable
}

void carouselPlacementGate() {
  CarouselPlacementGate gate;
  float const frame = 1.0F / 60.0F;

  // Closed: never places, and opening starts with a required release.
  CHECK(!gate.update(false, 0, true, true, true, true, frame));

  // Pressing while the centre is still loading is consumed. Holding through
  // the eventual load cannot place the model that appeared late.
  CHECK(!gate.update(true, 0, false, true, true, true, frame));
  CHECK(!gate.ready());
  for (int i = 0; i < 60; ++i) {
    CHECK(!gate.update(true, 0, true, true, true, false, frame));
  }
  CHECK(!gate.ready()); // still the same held press: release required first

  // Release. The settle dwell is already past, so the next fresh press places.
  CHECK(!gate.update(true, 0, true, true, false, false, frame));
  CHECK(gate.ready());
  CHECK(gate.update(true, 0, true, true, true, true, frame));
  CHECK(!gate.update(true, 0, true, true, true, true, frame)); // no repeat

  // Opening/closing resets: the settled, loaded centre needs a fresh press.
  gate.reset();
  CHECK(!gate.ready());
  CHECK(!gate.update(true, 0, true, true, true, true, frame));

  // A moving ring is never armed, even with a loaded centre.
  gate.reset();
  for (int i = 0; i < 30; ++i) {
    CHECK(!gate.update(true, 0, true, false, false, false, frame));
  }
  CHECK(!gate.ready());
  CHECK(!gate.update(true, 0, true, false, true, true, frame));
  CHECK(!gate.ready());
  // It arms only after the ring rests for the dwell.
  CHECK(!gate.update(true, 0, true, true, false, false, frame));
  CHECK(!gate.ready());
  for (int i = 0; i < 8; ++i) {
    gate.update(true, 0, true, true, false, false, frame);
  }
  CHECK(gate.ready());

  // Browsing to another index disarms; loading THAT index is required, and a
  // press during the transition never leaks into a placement later.
  CHECK(!gate.update(true, 1, true, false, true, true, frame));
  CHECK(!gate.ready());
  for (int i = 0; i < 30; ++i) {
    CHECK(!gate.update(true, 1, true, true, true, false, frame));
  }
  CHECK(!gate.ready());
  CHECK(!gate.update(true, 1, true, true, false, false, frame));
  CHECK(gate.ready()); // dwell from the stable frames above has elapsed

  // A different index being loaded does not satisfy the centre. Pass an index
  // whose slot is still loading: no arm, no place.
  gate.reset();
  for (int i = 0; i < 30; ++i) {
    CHECK(!gate.update(true, 2, false, true, false, false, frame));
  }
  CHECK(!gate.ready());
  CHECK(!gate.update(true, 2, false, true, true, true, frame));

  // Non-finite frame times never arm the dwell.
  gate.reset();
  for (int i = 0; i < 10; ++i) {
    (void)gate.update(true, 0, true, true, false, false,
                      std::numeric_limits<float>::quiet_NaN());
  }
  CHECK(!gate.ready());
  (void)gate.update(true, 0, true, true, false, false, -1.0F);
  CHECK(!gate.ready());
}

void carouselStickAvailability() {
  constexpr float frame = 1.0F / 60.0F;
  Vec2 const neutral{0.0F, 0.0F};
  Vec2 const held{0.8F, 0.0F};
  float const nan = std::numeric_limits<float>::quiet_NaN();
  float const infinity = std::numeric_limits<float>::infinity();
  CHECK(!carouselStickNeutral(false, neutral));
  CHECK(carouselStickNeutral(false, neutral, true)); // confirmed simple controller
  CHECK(!carouselStickNeutral(true, held, true)); // active binding still must be neutral
  CHECK(!carouselStickNeutral(true, {nan, 0.0F}, true));
  CHECK(carouselStickNeutral(true, neutral));
  CHECK(carouselStickNeutral(true, {0.5F, 1.0F}));
  CHECK(!carouselStickNeutral(true, held));
  CHECK(!carouselStickNeutral(true, {nan, 0.0F}));
  CHECK(!carouselStickNeutral(true, {0.0F, infinity}));

  // A confirmed stick-less controller can place, but cannot bypass loading,
  // rest, settle dwell, or the consumed-press release requirement.
  CarouselPlacementGate noStick;
  bool const available = carouselStickNeutral(false, neutral, true);
  CHECK(!noStick.update(true, 0, false, available, false, false, frame));
  CHECK(!noStick.update(true, 0, false, available, true, true, frame));
  for (int i = 0; i < 30; ++i) {
    CHECK(!noStick.update(true, 0, true, false, true, false, frame)); // moving
  }
  CHECK(!noStick.ready());
  for (int i = 0; i < 30; ++i) {
    CHECK(!noStick.update(true, 0, true, available, true, false, frame));
  }
  CHECK(!noStick.ready()); // loading press was consumed
  CHECK(!noStick.update(true, 0, true, available, false, false, frame));
  CHECK(noStick.ready());
  CHECK(noStick.update(true, 0, true, available, true, true, frame));

  CarouselPlacementGate gate;
  auto update = [&](bool active, Vec2 stick, bool triggerHeld, bool pressed) {
    return gate.update(true, 0, true, carouselStickNeutral(active, stick),
                       triggerHeld, pressed, frame);
  };
  for (int i = 0; i < 30; ++i) {
    CHECK(!update(true, neutral, false, false));
  }
  CHECK(gate.ready()); // loaded centre is settled
  CHECK(!update(false, neutral, true, true)); // unavailable is not neutral
  CHECK(!gate.ready());
  // The held trigger cannot place when the stick returns neutral.
  for (int i = 0; i < 30; ++i) {
    CHECK(!update(true, neutral, true, false));
  }
  CHECK(!gate.ready());
  CHECK(!update(true, neutral, false, false));
  CHECK(gate.ready());
  CHECK(update(true, neutral, true, true));

  // Reacquiring a deflected stick after an interruption must not arm.
  gate.reset();
  for (int i = 0; i < 30; ++i) {
    CHECK(!update(false, neutral, false, false));
  }
  CHECK(!update(true, held, true, true));
  CHECK(!gate.ready());
  for (int i = 0; i < 30; ++i) {
    CHECK(!update(true, held, true, false));
  }
  CHECK(!gate.ready());
  for (int i = 0; i < 30; ++i) {
    CHECK(!update(true, neutral, true, false));
  }
  CHECK(!gate.ready());
  CHECK(!update(true, neutral, false, false));
  CHECK(gate.ready());
  CHECK(update(true, neutral, true, true));
}

void viewTrackingTriggerInterruption() {
  SelectTriggerLatch latch;
  CHECK(!latch.updateTracked(true, true, 0.0F).pressed);
  CHECK(latch.updateTracked(true, true, 1.0F).pressed);
  auto lost = latch.updateTracked(false, true, 0.0F);
  CHECK(!lost.active && lost.held && !lost.released);
  CHECK(!latch.updateTracked(false, true, 1.0F).pressed);
  CHECK(!latch.updateTracked(true, true, 1.0F).pressed);
  CHECK(!latch.updateTracked(true, true, 0.0F).released);
  CHECK(latch.updateTracked(true, true, 1.0F).pressed);
}

void controllerTriggerWorkflow() {
  using T = ControllerTriggerWorkflow::Target;
  using A = ControllerTriggerWorkflow::Action;
  ControllerTriggerWorkflow workflow;

  // A selection that already existed at construction is not primed.
  ControllerTriggerWorkflow existing;
  CHECK(existing.update(true, true, true, true, T::selected, true) == A::select);
  CHECK(existing.update(true, true, true, true, T::selected, true) == A::none);
  CHECK(existing.update(true, false, false, true, T::selected, true) == A::none);
  CHECK(existing.update(true, true, true, true, T::selected, true) == A::beginGrab);

  // The first press selects and is consumed: holding it cannot grab.
  CHECK(workflow.update(true, true, true, true, T::other, false) == A::select);
  CHECK(workflow.update(true, true, false, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::none); // held
  CHECK(workflow.update(true, false, false, true, T::selected, true) == A::none);

  // Release, then the next press on the current selection grabs.
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::beginGrab);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::selected, true) == A::none);

  // A different model still selects instead of grabbing, even when primed.
  CHECK(workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::other, true) == A::select);
  CHECK(workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::none); // held
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);

  // A tracked ray miss grabs without pointing at the selection; a hold
  // cannot begin a second grab.
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::beginGrab);
  CHECK(workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);

  // Missing aim is not a miss: consume the press even when primed, and do
  // not replay it when tracking returns until a real release and new press.
  CHECK(workflow.update(true, true, true, false, T::none, true) == A::none);
  CHECK(workflow.primed() && workflow.blocked());
  CHECK(workflow.update(true, true, false, true, T::other, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::beginGrab);
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);

  // With no primed selection, unavailable aim cannot clear a selection or
  // select a target, and the same held press cannot act after reacquisition.
  ControllerTriggerWorkflow unavailable;
  CHECK(unavailable.update(true, true, true, false, T::none, true) == A::none);
  CHECK(unavailable.update(true, true, false, true, T::other, true) == A::none);
  CHECK(unavailable.update(true, false, false, false, T::none, true) == A::none);
  CHECK(unavailable.update(true, true, true, false, T::other, true) == A::none);
  CHECK(unavailable.update(false, false, false, true, T::none, true) == A::none);
  CHECK(unavailable.blocked()); // inactive release cannot re-arm
  CHECK(unavailable.update(true, true, true, true, T::other, true) == A::none);
  CHECK(unavailable.update(true, false, false, true, T::none, true) == A::none);
  CHECK(unavailable.update(true, true, true, true, T::other, true) == A::select);

  // With no primed selection, a genuine miss still clears or does nothing.
  ControllerTriggerWorkflow missed;
  CHECK(!missed.primed());
  CHECK(missed.update(true, true, true, true, T::none, true) == A::clearSelection);
  CHECK(!missed.primed());
  CHECK(missed.update(true, false, false, true, T::none, false) == A::none);
  CHECK(missed.update(true, true, true, true, T::none, false) == A::none);

  // Removing the selection invalidates priming before handling a miss.
  CHECK(workflow.update(true, false, false, true, T::none, false) == A::none);
  CHECK(!workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::none, false) == A::none);
  CHECK(workflow.update(true, false, false, true, T::none, false) == A::none);

  // A disabled workflow makes no decision, including no clear on a miss.
  CHECK(workflow.update(false, true, true, true, T::none, true) == A::none);
  CHECK(workflow.update(false, false, false, true, T::none, true) == A::none);

  // A pre-existing selection must be selected and released before grabbing.
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::select);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::beginGrab);

  // A tracking interruption clears priming and requires a real release.
  workflow.reset();
  CHECK(workflow.blocked());
  CHECK(!workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::selected, true) == A::none);
  CHECK(!workflow.blocked());
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::select);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::selected, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::beginGrab);

  // A reset prevents grabbing on a miss until a new selection is made.
  workflow.reset();
  CHECK(!workflow.primed());
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::none);
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::none, true) == A::clearSelection);

  // External selection removal invalidates prior priming on the same frame.
  CHECK(workflow.update(true, false, false, true, T::none, false) == A::none);
  CHECK(workflow.update(true, true, true, true, T::selected, true) == A::select);
  CHECK(workflow.primed());
  CHECK(workflow.update(true, false, false, true, T::none, true) == A::none);
  CHECK(workflow.update(true, true, true, true, T::none, false) == A::none);
  CHECK(!workflow.primed());
}

void toolPanelPress() {
  using A = ToolPanelPress::Action;
  ToolPanelPress press;
  float const frame = 1.0F / 60.0F;
  CHECK(!press.active());
  CHECK(!press.ownsTrigger());

  // A normal button fires on release over the same button only.
  press.press(ToolPanelPress::kModels);
  CHECK(press.active() && press.ownsTrigger());
  CHECK(press.hold(true, ToolPanelPress::kModels, .1F) == A::none);
  CHECK(press.release(ToolPanelPress::kRedo) == A::none); // slid off: cancel
  CHECK(!press.active());

  press.press(ToolPanelPress::kUndo);
  CHECK(press.release(ToolPanelPress::kUndo) == A::undo);
  CHECK(!press.active());

  // Sliding off then back cannot reactivate a normal button, and a
  // cancelled capture still owns the trigger until release.
  press.press(ToolPanelPress::kUndo);
  CHECK(press.hold(true, ToolPanelPress::kRedo, frame) == A::none);
  CHECK(press.active() && press.ownsTrigger());
  CHECK(press.hold(true, ToolPanelPress::kUndo, frame) == A::none);
  CHECK(press.release(ToolPanelPress::kUndo) == A::none);
  CHECK(!press.active());

  // A miss is captured (so it cannot click through) and fires nothing.
  press.press(-1);
  CHECK(press.ownsTrigger());
  CHECK(press.release(-1) == A::none);
  CHECK(!press.active());

  // Delete is a 0.6 s hold; it fires once, then stays captured until release.
  press.press(ToolPanelPress::kDelete);
  for (int i = 0; i < 20; ++i) {
    CHECK(press.hold(true, ToolPanelPress::kDelete, frame) == A::none);
  }
  CHECK(press.charge() > .4F && press.charge() < .6F);
  bool fired = false;
  for (int i = 0; i < 30; ++i) {
    if (press.hold(true, ToolPanelPress::kDelete, frame) == A::deleteNow) {
      fired = true;
    }
  }
  CHECK(fired);
  CHECK(press.charge() == 1.0F);
  CHECK(press.hold(true, ToolPanelPress::kDelete, frame) == A::none); // only once
  CHECK(press.release(ToolPanelPress::kDelete) == A::none);
  CHECK(!press.active());

  // Delete cannot charge away from its button, even if the pointer returns.
  press.press(ToolPanelPress::kDelete);
  CHECK(press.hold(true, ToolPanelPress::kDelete, .3F) == A::none);
  CHECK(press.charge() > 0.0F);
  CHECK(press.hold(true, -1, .4F) == A::none);
  CHECK(press.charge() == 0.0F);
  CHECK(press.ownsTrigger());
  CHECK(press.hold(true, ToolPanelPress::kDelete, 1.0F) == A::none);
  CHECK(press.charge() == 0.0F);
  CHECK(press.release(ToolPanelPress::kDelete) == A::none);

  // Releasing early cancels; a tracking loss cancels; charge never fires.
  press.press(ToolPanelPress::kDelete);
  (void)press.hold(true, ToolPanelPress::kDelete, .3F);
  CHECK(press.release(ToolPanelPress::kDelete) == A::none);
  CHECK(!press.active());
  press.press(ToolPanelPress::kBackdrop);
  press.cancel();
  CHECK(!press.active());

  // A new press after cancellation is independent and may fire normally.
  press.press(ToolPanelPress::kDelete);
  CHECK(press.hold(true, ToolPanelPress::kDelete, .6F) == A::deleteNow);
  CHECK(press.release(ToolPanelPress::kDelete) == A::none);

  // Non-finite/negative frame times cannot charge or fire.
  press.press(ToolPanelPress::kDelete);
  (void)press.hold(true, ToolPanelPress::kDelete, std::numeric_limits<float>::quiet_NaN());
  CHECK(press.charge() == 0.0F);
  (void)press.hold(true, ToolPanelPress::kDelete, -1.0F);
  CHECK(press.charge() == 0.0F);
  CHECK(press.release(ToolPanelPress::kDelete) == A::none);

  // Actions map exactly to the five real buttons.
  CHECK(ToolPanelPress::actionFor(ToolPanelPress::kModels) == A::models);
  CHECK(ToolPanelPress::actionFor(ToolPanelPress::kUndo) == A::undo);
  CHECK(ToolPanelPress::actionFor(ToolPanelPress::kRedo) == A::redo);
  CHECK(ToolPanelPress::actionFor(ToolPanelPress::kDelete) == A::deleteNow);
  CHECK(ToolPanelPress::actionFor(ToolPanelPress::kBackdrop) == A::backdrop);
  CHECK(ToolPanelPress::actionFor(-1) == A::none);
}
} // namespace

int main() {
  HandSelectionLatch latch;
  HandGesture deliberate;
  CHECK(!deliberate.update(true, 0));
  CHECK(deliberate.update(true, 1));
  CHECK(!latch.startsGrab(true)); // first press selects, never moves
  latch.selected(true);
  CHECK(!deliberate.update(true, 1)); // a held first press cannot grab
  CHECK(!deliberate.update(true, 0));
  CHECK(deliberate.update(true, 1));
  CHECK(latch.startsGrab(true));
  CHECK(!latch.startsGrab(false)); // a different hit selects instead
  latch.reset();
  CHECK(!latch.startsGrab(true));
  HandPose const origin{{0,0,0}, {}, true};
  HandPose const tiny{{.001F,0,0}, {}, true};
  auto const smoothed = smoothHandPose(origin, tiny, .01F);
  CHECK(smoothed.position.x > 0 && smoothed.position.x < .001F);
  CHECK(smoothHandPose(smoothed, tiny, .01F).position.x > smoothed.position.x);
  CHECK(handScaleRatio(.2F, .4F) == 2.0F);
  CHECK(handScaleRatio(.0F, .4F) == 1.0F);
  leftTriggerWorkflow();
  booleanRearmLatch();
  togglePressInterruptions();
  historyPressInterruptions();
  deleteContextInterruptions();
  selectTriggerLatch();
  graspArbitration();
  contactCacheLifetime();
  surfaceContact();
  gestures();
  selectionGraceAndTracking();

  carouselActions();
  panels();
  palms();
  projection();
  grabs();
  controllerPoseQuality();
  carouselPlacementGate();
  carouselStickAvailability();
  viewTrackingTriggerInterruption();
  controllerTriggerWorkflow();
  toolPanelPress();
  std::printf("Hand tests: %d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
