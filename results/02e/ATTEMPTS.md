# 02E execution notes

## Phase 0: reproducibility before anything new

- **Published evidence verified.** 02B (103 files; 20 local-only files absent and
  reported as unverifiable), 02C and 02D (287 hashes) pass their own verifiers.
- **02D evidence audit recomputed from saved data:** 2,208 checks. 02D's
  `validate02d` CLI rewrites `results/02d/validation.json`, so its pure `audit()`
  was called directly; nothing under `results/02d` changed.
- **Fresh numerical reproduction.** 02D's unchanged preflight reproduced all 120
  historical 02C outcomes, plus zero-residual exactness, determinism, passive
  recording and BAM parameters. All 1,044 02D held-out rows (612 grid, 432
  shift) were re-simulated, and every row equals its published row exactly (6
  workers, 922 s).

### The compiled-model hash mismatch

- The historical `model.mjb` (`ebb2ce…`) was compiled on **Windows**: MSVC
  Python, upstream at `C:\Users\James\microduck_rl`. Recompiling there with the
  same packages reproduces `ebb2ce…` exactly. Recompiling on Linux/WSL gives
  `cba1a6…`, the hash 02D reported. Each platform is deterministic.
- `model.json` records upstream XML/constants hashes that Linux appears to
  contradict. Those are the CRLF hashes of a `core.autocrlf=true` Windows
  checkout: converting the Linux LF files to CRLF reproduces them byte for byte.
  Upstream content is identical at `b02fdea`, clean.
- Field by field, the two binaries agree in 584 of 590 numeric model fields:
  masses, inertias, geometry sizes, solver options, mesh vertices, faces and
  normals, names, asset paths. All six differing fields are MuJoCo
  convex-hull/polygon arrays (`mesh_graph`, `mesh_poly*`). In 8 of 38 meshes
  their bytes differ, including both soles.
- Every hull has the same vertex set. Six have identical face and polygon sets
  and differ only in ordering. `jaw` triangulates coplanar facets differently;
  `power_support` has a different polygon set.
- A minimal identical MJCF around a single upstream STL (`platform02e`) gives the
  same vertices but different hull/polygon bytes on Windows vs Linux MuJoCo
  3.10.0. The cause is platform-dependent hull processing in MuJoCo's compiler,
  not upstream drift, line endings or mjlab.
- Dynamics: on the fresh Linux binary, 1,009 of 1,044 02D trials are bitwise
  identical and **all 1,044 outcomes are identical**. Every one of the 35
  divergences starts after a fall, non-foot contact or self-contact (earliest
  1.67 s), when ordering-dependent contacts on the shells begin.
- **Decision:** the hash guard was not weakened. 02E simulates only with the
  hash-matching historical binary. The Linux binary stays local evidence; it
  embeds upstream meshes. `fields_digest_excluding_hull` is a diagnostic identity,
  not a replacement guard.

## Feasibility before the protocol (exploratory; results in `feasibility/`)

- `sim02e` with no 02E intervention is bit-identical to `sim02d`, with and
  without a +2° offset.
- Static gravity-vs-leg-kinematics, level floor, reflex standing: estimates of
  +1.93/−2.01/+2.17° for +2/−2 pitch and +2 roll; 0.11/−0.06° at nominal. A 2°
  pitch slope read as 2.08°, and +2 offset on a +2 slope as 3.89°. This is the
  predicted slope confound.
- Hip-yaw probe (both hip yaws, either sign, 0.05–0.2 rad): with both feet
  planted the trunk does not yaw cleanly; the feet twist or slip (up to 5 mm at
  0.2 rad, 10.6° tilt). Estimates were noise.
- Rocking probe, first formulation (per-step rotation increments): signal
  ~0.0025 rad per step vs comparable encoder-difference noise; errors 4–150°.
- Second formulation (cumulative rotation vectors): still 5–40°, and the large
  false corrections then toppled the reflex. A noise-free diagnosis showed the
  gyro integrates trunk rotation to ≤0.002 rad. The leg-kinematic trunk rotation
  was wrong by 0.007–0.043 rad because the soles rotated 0.7–3.7°. The signal
  (offset × probe rotation) is second order, about 0.001 rad.
- Before declaring the protocol, the probe estimator was corrected to compare
  angular velocities, with a single-axis fit for a single-axis probe. The
  cumulative formulation had scored the unobservable roll component and carried
  a second-order integration error. A smoke test of the corrected estimator:
  clean-gyro oracle exact at nominal; noisy-gyro oracle 0.03–1.0°;
  kinematic-from-true-joints 0.7–14° (rate error 35–89% from sole rotation);
  the 0.10 rad probe fell in all three smoke trials.
