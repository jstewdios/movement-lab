# Bounded 01B departure plan

Maximum eight independent six-second GPU rollouts, seed 42, checkpoint99,
curated groundcontact, root z reset 0.1181823627 m / approximately 1 mm sole clearance.
200 Hz refreshed endpoint recording and CPU contact audit, 50 Hz actor/commands.
Same BAM settings, no training, upstream changes or post-reset state writes.

Order: zero baseline; head yaw ±0.08 rad; bilateral flex positive (hip pitches
∓0.03, knees ±0.05, ankles ∓0.02 rad); lateral hip-roll shift ±0.025 rad;
unilateral left flex (hip −0.02, knee +0.035, ankle −0.015 rad); final zero control.
Each single-run subprocess resets independently. Prefixes before 2 s must match
the zero baseline. Extra variants or repeats are not authorized by this plan.

Cosine pulse: start 2 s; rise to 2.6 s; hold until 3 s; return by 3.6 s;
observe through 6 s. Default target + actor + residual; no clipping silently
changes this composition. Named-joint composed targets retain 0.02 rad limit margin.

Before any departure, validate the zero baseline under the original six-second
baseline gate. Each run additionally checks the pre-command [1.5,2) s settling
window, and stops on >3 mm penetration, nonfoot ground contact, >12° tilt,
nonfinite state, force/speed/joint-limit diagnostic bounds or termination.
Stops occur at the next control boundary, up to 20 ms later. No later run is
scheduled after physical-gate or recovery failure.

Final [5.5,6] s must satisfy the same local stability criteria: both feet contact,
COM speed ≤0.03 m/s, angular speed ≤0.3 rad/s, horizontal per-axis COM range ≤5 mm,
tilt <12°, no nonfoot contact. Return relative to pre-command means requires
horizontal COM displacement ≤5 mm, height change ≤2 mm, orientation change ≤3°,
and every joint offset ≤0.03 rad, throughout that final window. Also require a
continuous 0.5 s stable/returned window after the residual ends; report its onset
delay separately from the duration needed to confirm it.

A controlled response additionally requires matched-control peak actual joint
response ≥0.005 rad or COM response ≥0.5 mm during [2,3.6) s. Every such detected
response channel must contract to less than half its peak in the final mean.
These exploratory thresholds distinguish articulation-and-return from merely
surviving. They do not certify hardware fidelity or robustness. Foot-force shifts,
contact transitions, joint targets/actuals and effort are reported continuously;
no stepping or unloading is inferred from target commands.
