# 01A preflight and diagnostic attempts

No 00A–00C records were changed. These directories remain local; none were excluded because of an undesirable physical outcome.

- `pilot`: CUDA driver unavailable inside sandbox; no trajectory.
- `pilot-gpu`: action/joint name-container equality assertion; no trajectory. The model has 14 actuated joints. An early approval prompt incorrectly attributed the mismatch to a passive jaw joint; XML inspection did not support that explanation. Explicit action-manager target IDs now define ordering.
- `smoke`: `mj_saveLastXML` cannot export a model assembled with MjSpec; no trajectory. Switched to compiled MJB state-replay model.
- `smoke-state`: one complete 3 s baseline, but CPU ground-contact counts are invalid because terrain is a fixed child body, not world body 0. Geometry penetration measurement and dynamics remain diagnostic; this record is not included in the atlas. Corrected by explicit terrain geom ID.
- `atlas`: one complete baseline with corrected contact audit; reset before the next sequence failed because BAM delay tensors allocated in PyTorch inference mode were reset outside it. Retained locally. Reset and stepping now share inference mode.
- `atlas-v2`: final intended nine-trial pilot; all physical outcomes retained. See report for completion and validation.

Preflight wall times were not recorded consistently and are excluded from the final pilot's measured cost. There was no RL training or adaptive search. Compiled models embed upstream geometry and remain local under existing upstream asset terms.
