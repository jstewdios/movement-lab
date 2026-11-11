# 02D execution notes

## Model restoration

The Linux checkout lacked the local-only compiled model. The unmodified 02B
regenerator first failed at the sandbox's read-only Warp cache, then compiled
with the matching upstream revision/packages after escalation. The binary hash
was `cba1a6dab0683d8ecaa26872b573c48a82bf9c27b7c260c25e80d74abcf0b9bd`,
different from the historical `ebb2ce87cae95c606e6db4ad2e790e9f4309e557443d6ea1a0629d5ba5e3cf8c`.
The existing hash guard rejected it; nothing was installed. No tolerance or
historical metadata was changed. The exact preserved binary was found at
`/mnt/c/Users/James/movement-lab/results/02b/model/model.mjb`, hash-checked,
and copied into the Linux checkout. Its assets remain local, as in 02B/02C.
The cause of the alternate compile hash was not established.

`preflight.json` then reproduced all 120 declared historical outcomes exactly,
including seed 2's small-push failures and seed 1's 80 ms delay failures. The
new zero-residual rollout has identical qpos, qvel and torques; repeated records
are deterministic and the measurement copy leaves dynamics unchanged.
The 22-parameter controller averaged 26 microseconds per call in the local
10,000-call CPU measurement, below the 20 ms update period. This is a software
benchmark, not a hardware real-time guarantee.

## First pilot: useful recovery signal, unacceptable quiet regression

Seed 71 used 1,536 episodes in 168.8 s. Development validation found 21/24
recoveries (frozen reflex 16/24; 02C oracle and fixed IMU trigger 20/24 each).
The learned controller gained six paired recoveries but fell in quiet seed
5,000,000,003 at 2.52 s. This was a real tilt failure with later nonfoot contact,
not an unsupported numerical endpoint. The quiet gate correctly stopped Phase B.
Training's final winner had also lost a small-push case. The bounded revision
in PROTOCOL.md increases quiet/benign coverage while preserving episode count.
No earlier successful-looking generation was substituted for the final winner.

## Revised pilot: gate passed, confirmation authorized by the protocol

Seed 72 used the same 1,536 episodes in 208.5 s, with 16 candidates × 12 strata
instead of 24 × eight. Its final training winner recovered all 12 final-batch
conditions. On the fresh development seeds it recovered 20/24 versus 16/24 for
the frozen reflex, 19/24 for the 02C oracle, and 17/24 for the fixed IMU trigger.
Four paired gains and zero losses passed the original gate, including all four
quiet trials. This is a small development screen, not final confirmation.
Phase B then launched both arms at the declared 2,304-episode budget per seed.

## Supplementary provenance check

While the confirmation jobs ran, the installed BAM parameter file was checked
against 02C: both its raw and canonical hashes match exactly. Upstream remains
clean at `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`. The machine/Python details
and result are saved in `provenance.json`. The preflight command was extended
to make this motor-parameter check mandatory on future reproductions; the
original preflight outcomes remain unchanged. No new GPU cross-check is claimed.

## Control-definition correction before final evaluation

The rollout's generic event time also identifies sensor-only transients for
auditing. A review found that passing it directly to the 02C oracle would give
that control an erroneous onset cue during a false alarm. The evaluation wrapper
now omits the fixed option when there is no physical push or force pulse; its
standing reflex still receives the same corrupted IMU. The fixed IMU trigger
remains eligible to fire. This preserves the meaning of a true physical-onset
oracle. It changes no learner or training source, and neither pilot validation
battery contained sensor-only transients, so their numerical results are
unchanged. Their evaluator source versions are preserved in `pilot/source-v1`
and `pilot/source-v2`. A physics regression test checks this distinction.

The staged driver also pins reused optimizer/model/measurement helpers in
`dependencies.json`, supplementing each trainer's source hashes. These helpers
were verified unchanged from the base commit before the inventory was saved.
Future resumes through the driver reject changed helpers; this is a provenance
guard only and changes no numerical training operation.
