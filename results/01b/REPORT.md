# Movement Lab 01B — Stable Baseline and Controlled Departures

**Stopped at the baseline contact gate. No validated starting condition and no controlled-departure trials.** This is a failed baseline-validation attempt, not evidence that a standing controller cannot work with groundcontact. No movement transition, recovery, stepping, or stable endpoint was demonstrated.

## Baseline investigation

The completed 01A report, saved sources, provenance, validation, raw zero trajectory and compiled replay were inspected alongside 00A validation and 00B–00C reports and evaluator sources. A fresh CPU audit of historical 01A zero replay gives maximum link-position error 3.17e-8 m. Earlier records and upstream remain untouched.

01A zero and 00A checkpoint-99 initial generalized positions and velocities match **exactly**. Both use upstream HOME (STAND2), root reset z offset +0.125 m, zero x/y/yaw offsets and zero initial velocity. Both initially descend without support. There was no independent stable initialization in either: 00A's actor arrests the landing and maintains standing; 01A's zero action holds HOME through powered actuators and falls. 00B default-position controls also could not establish standing. 00C recovery results inherit the walk model and do not validate body-ground collision behavior.

| Property | 00A standing | 01A | 01B attempted baseline |
|---|---|---|---|
| Controller | checkpoint actor with observation normalizer/history | zero raw action, powered HOME targets | checkpoint 99 actor with upstream normalizer/history |
| Initial state | HOME, +0.125 m reset offset, zero velocity | identical | identical nominal reset |
| Collision model | walk | curated groundcontact | curated groundcontact retained |
| Action mapping | default position + actor action, scale 1 rad/unit | default position + named offsets | default position + actor + residual; residual exactly zero in attempted trial |
| BAM | XL330 M6; firmware kp 200; 7.35 V; sag 0.1; floor 6 V; fixed 4-substep delay; friction and torque constraints retained | same | same |

The XML audit finds identical masses (0.73724318 kg), inertia tensors/frames, body transforms, joint axes/ranges/damping/armature and actuator gears. Joint spaces remain nq=21, nv=20, nu=14. The validated policies are from velocity/walk training, not groundcontact training; only checkpoints 99/348 have validated standing evidence in this repository. 99 was chosen for the smaller prior standing effort, not presumed transferable success. No compatible groundcontact standing checkpoint was identified among the local velocity runs.

Walk XML has two terrain-enabled sole geoms and three self-only geoms (power support and two legs). Groundcontact retains power support as self-only, changes both legs to terrain-enabled, and adds six terrain-enabled meshes: battery, two hips, top head shell, jaw and bottom head shell. Thus raw active collision geoms increase from 5 to 11. Terrain-enabled geoms increase from 2 to 10. Neither is a complete whole-robot collision model. FULL_COLLISION applies by **name**: named soles have condim 3, priority 1 and friction 1. The added unnamed mesh geoms retain XML defaults (condim 3), rather than automatically receiving the named `.*_collision` condim-1 override. Collision bit masks and per-geom details are in [model-audit.json](model-audit.json). No sole geometry was substituted and no collision geoms were disabled.

## Declared budget and gate

The budget announced before evaluation was at most ten 6-second GPU trials, seed 42: baseline, repeat, then four signed residual pairs (head yaw ±0.08 rad, bilateral flexion up to ±0.05 rad, hip-roll shift ±0.025 rad, unilateral flexion up to ±0.035 rad). Feedback remains active; commands would start at 2 s, return by 3.6 s and allow 2.4 s recovery. These are unexecuted plans, not observations.

The baseline requires a full 6 s record, finite state, no nonfoot contact, tilt below 12°, penetration no deeper than **3 mm at any sample**, and, from 1.5 s onward, both feet in contact, COM speed at most 0.03 m/s, root angular speed at most 0.3 rad/s and per-axis horizontal COM range at most 5 mm. The 3 mm limit is a deliberately conservative exploratory contact criterion, declared in source before the run; it is not a calibrated simulator or hardware tolerance. It is stricter than 01A's 10 mm invalidity limit. Limits were not relaxed after failure.

The sandbox attempt could not initialize the CUDA driver and integrated zero steps. The authorized outside-sandbox GPU attempt completed startup and recorded **12 physics samples / 0.060 s** before stopping at the next 20 ms control boundary. The first violating sample was at 0.050 s: terrain/right sole penetration 3.126 mm, peaking at **3.160 mm at 0.055 s**. The offending pair was independently identified in CPU forward geometry, not inferred from foot occupancy. This is a landing transient before any settling assessment or residual command. No nonfoot ground contact occurred.

| Measured baseline quantity (short landing interval only) | Value |
|---|---:|
| Maximum trunk tilt | 0.849° |
| Horizontal COM excursion from reset | 0.550 mm |
| COM vertical displacement at stop | −9.977 mm |
| Foot-contact sample occupancy L / R | 41.7% / 41.7% |
| Final COM speed | 0.0470 m/s |
| Final root angular speed | 0.739 rad/s |
| Peak / RMS endpoint actuator joint effort | 0.2470 / 0.0959 Nm |
| Maximum joint tracking error | 0.3419 rad |
| Largest actual joint range | 0.0640 rad |
| Sampled absolute mechanical work proxy | 0.03393 J |

These values cannot characterize steady standing. No repeat or departure was started. The available controller was not shown to settle safely under this contact criterion. The failure involves the unchanged sole, so it does **not** isolate an adverse effect of the added collision meshes or prove policy incompatibility. 00A did not apply this penetration gate; its standing success cannot certify it passes this criterion either.

A lower, contact-aligned static initial pose is a reasonable future baseline candidate, but a static pose or COM projection alone cannot establish compliant-contact equilibrium with BAM delay/friction. No such candidate was dynamically validated. In accordance with the instruction to stop on contact-validity failure, this run did not search new poses, switch checkpoints, retune contact parameters, loosen tolerances, pin the root, train a policy, or continue into movement discovery. Establishing a safe initial landing or supported reset remains prerequisite work.

## Evidence, replay and verification

[Measured replay](gpu-gate/replay.html) shows the **failed baseline only**, with saved link centers and actual/target joints. Its sparse two-projection visualization is not collision rendering. [Comparison plots](gpu-gate/atlas.png) describe the short transient. Full geometry replay uses `python -m movement_lab.replay01a results/01b/gpu-gate/zero`; it sets saved generalized states and forwards kinematics without integrating dynamics. Local full-rate NPZ and compiled MJB are retained and ignored by Git.

The GPU attempt consumed 23.17 s runner wall time including startup and 1.67 s within the trial. This is not GPU kernel time; sandbox failed-start time and offline analysis are additional. No training ran. Provenance records checkpoint hash, upstream revision/status and file hashes, GPU/software versions, resolved configuration and source/model hashes. A separate manifest covers this study's evidence and failed initialization logs locally.

All **25 tests pass**, including new signed-envelope and gate negative controls (quiet fallen pose, nonfoot support, penetration, unloading, COM velocity and incomplete horizon). Offline checks verify synchronized finite arrays, unit quaternions, zero initial velocity, actual default+actor+residual composition, named-joint generalized-state mapping, command schedule and gate recomputation. Independent CPU forward checks at **every** recorded sample reproduce link positions within 1.78e-8 m, COM within 1.25e-8 m and foot occupancy exactly. They reproduce contact distance as geometry, not GPU force-solver equivalence. Hash validation verifies original runner snapshots and checkpoint.

The inherited `summary.valid` field and generic atlas labels use 01A's 10 mm descriptive validity convention. They do **not** represent the 01B decision: `metadata.gate.pass=false` is authoritative. Endpoint forces/work are refreshed endpoint estimates, not integrated impulses, electrical energy or thermal measurements. Extra forward solves may alter warm starts, as documented in 00A. No stability or recovery classification is inferred from this truncated trace.

Revalidate without rerunning dynamics:

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m movement_lab.validate01b results/01b/gpu-gate
$PY -m unittest discover -s tests -v
```

The scientific result is limited to **unsupported initialization → sole contact with excessive penetration under the declared gate**. The research question about controlled departures remains unanswered. All earlier evidence is preserved; no commit, push, merge or 01C continuation was performed.
