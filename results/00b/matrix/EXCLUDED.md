# Excluded validation matrix

Do not use these trials for the checkpoint comparison. The seed-43 control
revealed an upstream command sampler override: `rel_forward_envs=0.2` could set
vx=0.3 even though the configured velocity range was (0,0). Both policies then
terminated before the intervention. Recorded command arrays exposed the issue.
The interrupted matrix is retained to document the validation failure.

The corrected study is in `../standing/`; 00B alone explicitly disables this
forward command bucket. The 00A code path and results remain preserved.
