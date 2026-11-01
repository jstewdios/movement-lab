# 02A execution notes

The sandbox's initial GPU availability query failed NVML initialization. No GPU simulation was attempted there. The authorized GPU pilot ran outside the sandbox with the prepared 600-second subprocess deadline, completing a baseline and two proposals per arm in 364.61 seconds. No simulation was repeated or excluded and no physical gate failed.

The offline proof first revalidated the four historical 01C movement records. A test initially exposed that the first version of the proposal score could not change its second choice based on a single observed validity label. Before any GPU trial, the score was changed to favor half-unit mutations near eligible outcomes and distance from failed neighborhoods; the history-sensitivity regression then passed. This was an offline implementation correction, not a post-result search change.

The executed explorer, driver and adapter are preserved under pilot-gpu. After the pilot, the adapter CLI was restricted to output path, coefficient pair and seed, preventing users from overriding declared timing/amplitude defaults. The pilot already used those fixed defaults. The final driver moves the historical baseline comparison into its pre-departure path and returns nonzero on hard failures. The new baseline had been independently verified during this pilot to match all historical 01C arrays, model and configuration. No outcome or search rule was changed after GPU observations.

There is one search seed, two proposals per arm, no reserved independent confirmation, and no evidence of a trained policy or discovery-efficiency improvement. The adaptive arm's second valid command was correctly rejected as a distinct discovery because measured phase-state differences were below threshold.

Earlier studies, source files and raw evidence remain unchanged. Local raw trajectories, compiled geometry and runtime logs remain hashed and preserved but excluded from Git. Nothing was committed, pushed, opened as a PR, or merged.
