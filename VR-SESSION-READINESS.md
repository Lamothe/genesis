# VR session readiness — what is verified unattended, and Michael's recipe

Status: 30/10/2026 (scope-pivot phase, item 3). Everything below was
verified unattended on platinum (AMD 8060S, RADV); the headset steps need
Michael at the machine.

## Verified unattended (with evidence)

1. **The OpenXR runtime resolves for native apps.** WiVRn runs as the
   flatpak `io.github.wivrn.wivrn` (server already active; NOT restarted).
   The loader needs the manifest pointer in the environment:

       export XR_RUNTIME_JSON=/var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/61d58e18b177b19a34f4a0f3063c571bfc7585ef8a3145327a5beb8b5e268332/files/share/openxr/1/openxr_wivrn.json

   With it set, `run-editor.sh --smoke-xr` gets PAST instance creation and
   runs, blocked waiting for a headset (verified: the process stays alive
   instead of failing with "no OpenXR runtime configured", which is what
   happens without the variable).

2. **The eye-view capture** (`run-editor.sh --capture-eyes DIR`, new):
   renders the scene from the two eye poses of a nominal head pose (the app
   camera offset by +/- half the standard 64 mm IPD) at the app's render
   FOV and writes eye-left.ppm / eye-right.ppm. Verified on the 8060S:
   18,446 vs 18,456 of 307,200 pixels covered - the two views differ (real
   parallax). This is what the headset would show as far as verifiable
   without one; the live session's eye poses and per-eye FOV come from the
   headset at run time. Evidence: ~/Projects/Genesis-dlss5/.build/capture/.

3. **The editor and its tests build and pass** on the worktree branch
   (859 checks + the neural-seam-native 29 checks).

## Michael's headset recipe (exact steps; what you should see)

1. Put the Quest on and turn it on. Open the WiVRn client app on the
   headset (installed via SideQuest or the WiVRn releases). It should
   show the server on this machine - select/pair it if prompted. First
   run may ask to accept the device on the server side.

2. On platinum (SSH or local), in a terminal:

       export XR_RUNTIME_JSON=/var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/61d58e18b177b19a34f4a0f3063c571bfc7585ef8a3145327a5beb8b5e268332/files/share/openxr/1/openxr_wivrn.json
       cd ~/Projects/Genesis-dlss5
       bash run-editor.sh ~/Downloads/VR/<some-model>.glb --xr

   (The spawn library at ~/Downloads/VR is scanned automatically; any
   .glb/.obj path works.)

3. **What you should see:** the headset view of the 3D scene - the
   carousel ring of models centred dead ahead at eye level, with the
   current model in the centre slot. Per the README: in VR the trigger
   only SELECTS (whole model, or a single part with grip+trigger); there
   are no flight controls by design (passthrough is the movement model).
   The scene is lit and rendered through Vulkan on the 8060S.

4. **What to verify:** head tracking moves the view; the carousel is
   level and centred; selecting with the trigger highlights the model or
   part; the app stays responsive (the renderer runs at the headset
   refresh via WiVRn). If the view is black: check the WiVRn client is
   connected (the server log on platinum shows a session) and that the
   terminal did not print an xrCreateSession error.

5. To end: close the window (or Ctrl+C the terminal); the headset returns
   to its home environment.

## The neural seam (the dlss5 boundary)

The dlss5 seam plumbing (two evaluator contexts behind
neural_seam_evaluate) is in this branch and its tests pass (29 checks).
Enabling it in the live VR session is the NEXT milestone (the headset
session is the validation boundary for the per-eye windowing decision in
dlss5-linux's vr/VR-SEAM-M1-ALIGNMENT.md).
