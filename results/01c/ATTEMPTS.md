# 01C attempts

The first `pilot` attempt could not initialize CUDA inside the filesystem sandbox
(CUDA error 100). No simulated sample or movement outcome was produced. The
partial configuration/source files and failure log remain local. This is
an infrastructure failure, not a failed physical baseline.

`pilot-gpu` retries the unchanged protocol with CUDA access outside the sandbox.
The trial driver stops before any movement if baseline simulation or its strict
saved-record/historical-prefix audit fails. No prior evidence is overwritten.

Full-rate trajectories and compiled upstream geometry remain local. Manifests
include their hashes; a checkout without those files cannot claim full validation.
