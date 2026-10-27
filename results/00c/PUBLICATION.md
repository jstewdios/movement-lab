# 00C publication audit

The publication set retains the runner and opt-in COM recorder, regression
tests, documentation, scientific report and interpretation, self-contained
comparison, two report-linked PNGs, all 44 retained trial metadata/initial
states/configurations/source snapshots, final analysis sources, adaptive
plan, measured costs, summaries, divergence metrics, validation results,
and complete-local-evidence SHA256 manifests.

Raw trial trajectories, evaluation logs, caches, eight redundant PNG exports,
and archived diagnostic/interrupted trial directories are ignored by Git.
They remain on disk. Exclusion notes and the archived-artifact manifest are
published under `excluded/`. Full manifests intentionally include unpublished
local evidence: missing files in a checkout are not evidence that those
records were validated there. See the repository README for regeneration
requirements and the excluded-record notes for the unresolved pre-event
mismatch and successful repeat. No scientific outcome was excluded for falling.

Publication verification reran all 18 unit/integration tests successfully,
validated all 44 retained trials with exact pre-event causal matching,
recomputed summary metrics exactly, verified complete-local-evidence and
checkpoint hashes, and checked the report's directional and seed-dependent
findings. The visualization contains 26 embedded figures and no scripts.
Configurations and training YAMLs match the preserved baseline provenance.
No additional GPU evaluations were launched for publication.

The 10,481,377-byte self-contained HTML is the largest published file; its
size is expected because it embeds the plots. Other PNG exports are redundant
with it. Credential-pattern scanning found no credentials; local paths and
username remain in provenance, as in the historical studies. Historical
00A/00B result files are unchanged. The only shared validator change repairs
short-trial indexing while retaining strict equality checks. Upstream Microduck
is clean at b02fdea4902c2b93b7d34a83a6f05da4ececb7d3.

Publication normalizes trailing whitespace in generated HTML/SVG, with no
change to plot tokens or scientific data. The generator, saved analysis
source, and evidence manifest include this formatting correction.
