# Preserved diagnostic records

`99-s43-dv+0.06-pre-mismatch` is excluded from boundary comparisons because
strict pre-intervention equality to its matched control failed. Maximum
pre-event differences were 1.18464e-5 in generalized position, 4.95514e-4
in generalized velocity, 6.35386e-5 in raw action, and 4.50648e-5 Nm in joint
actuator force. Contact flags matched exactly; COM position differed by at most
2.98210e-6 m. Other completed records matched exactly. These differences began
before injection, so they cannot be attributed to the disturbance. The cause
is unresolved; backend nondeterminism is a possibility, not an established
explanation. No tolerance was relaxed. A single repeat using identical settings
passed exact pre-event matching. Both attempts remain local.

`99-s42-dv-0.18-interrupted` was interrupted when the mismatch was discovered.
Its incomplete artifacts are preserved and excluded. Its replacement uses a
fresh directory under boundary/. The parent cost log contains completed
subprocess times including the excluded completed trial and its repeat, but
does not include the interrupted subprocess's elapsed time. No failed outcome
was removed because it fell: the exclusion criterion was pre-event mismatch.

The first terminated trials also exposed a validator indexing bug: a boolean
mask sized for a shorter disturbed record was applied to the longer control.
Integer sample indices now preserve the same strict comparisons on the shared
pre-event history. A synthetic short-record regression test also verifies that
pre-event changes still fail validation. Historical result files are unchanged.

Component units: root xyz mismatch was at most 2.41306e-6 m; joint position
mismatch 8.52346e-6 rad; body linear velocity mismatch 1.11918e-4 m/s; body
angular velocity mismatch 5.06826e-4 rad/s. The larger generalized-position and
body-velocity maxima above pool different coordinate types and must not be
assigned a single physical unit.
