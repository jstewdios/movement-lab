# PR #6 — P2 review fixes

No GPU trials or original reports, experimental records, snapshots or manifests
were regenerated. The original 3 mm penetration limit and physical gates remain
unchanged. Original source hashes still identify experimental code from commit
31b5ea29741314c075c184fe27e106c402be50a3; the separate review-fix manifest inventories
current implementations and new regression/validation evidence. Original inventory
entries are verified against that commit before accepting source amendments.

## Optional historical comparison

`historical02a` is shared by the study driver and report generator. It reports
`unavailable` if all historical artifacts are missing, `incomplete` if only some
are available, and `passed` only when trajectory, model and configuration are all
present and match. Every available artifact is compared strictly, even with partial
history. Mismatches record `failed` audit metadata and raise an error. Corrupt or
unreadable evidence also fails; missing new-study evidence is never optional.

On a fresh checkout the historical resolved configuration is normally present,
while raw arrays/model are absent: this is explicitly incomplete historical
validation. It does not prevent reporting new, independently validated 02A records.
New reports record availability, missing paths and comparison results in
`historical-baseline-audit.json`, `report-validation.json` and report prose. The
report generator runs the existing full per-trial validator against the new 02A
baseline, independent of history. The preserved original pilot report and its
historical comparison remain unchanged and continue to pass saved-record audit.

## Shared subprocess deadline and cleanup

The simulator and validator use the same absolute monotonic deadline through
`process02a.run_until`. Each runs in a dedicated Linux session/process group. On
timeout the whole group receives SIGKILL; direct children are drained and waited,
and Linux child-subreaper adoption permits waiting for nested descendants too.
Subreaper state is restored after cleanup. Normal/exceptional exits also clean up
remaining group members. Cleanup can take scheduling time beyond the deadline;
no simulator or validator work is intentionally allowed a grace interval past it,
and elapsed wall time includes cleanup. This requires Linux and covers the
inheriting subprocess tree used by the simulator and nested saved-record validator.

Interrupted attempts retain `deadline interrupted`, `interrupted_phase`, and
`subprocess_tree_reaped`; partial validator diagnostics are kept. They receive no
successful validation or archive outcome. A physically invalid completed experiment
remains a different case, handled by the unchanged physical-validation machinery.

CPU-only regressions exercise missing/partial/full/mismatched historical evidence,
fresh-checkout reporting and strict new-baseline validation. Real nested process
trees (including a SIGTERM-ignoring grandchild) are spawned for both execution and
validation timeouts. Each PID must be gone, not a surviving process or zombie; the
study's attempt/archive/cost files are checked. No GPU or upstream changes are used.

Validation: `tests.txt` records the full suite; `saved-record-validation.json`
recomputes the original 02A records and proposal histories; `earlier-validation.txt`
audits preserved 01C evidence. Run `python -m movement_lab.publication02a --full`
for both original and review inventories, including excluded local artifacts.

All 62 tests passed (the prior 56 plus six review regressions). Both saved-record
audits passed: all five original 02A trials retain their validation and discovery
results, and the earlier 01C physical/historical audit remains unchanged. The
supplemental inventory verifies current fixes alongside the immutable original
publication inventory and all 27 excluded local artifacts.
