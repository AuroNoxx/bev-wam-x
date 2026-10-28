# bev-wam-x v1 archive (2026-10-24)

This archive preserves modules moved out of the active tree during the v2 transition. See [v1 documentation](../../docs/V1.md) and [v2 documentation](../../docs/V2.md).

Correction: the earlier archive note described v1 as RGB/camera. Retained source (`models/latent.py::BEVEncoder`) explicitly consumes four-channel semantic BEV. The architectural difference is deterministic/separate candidate-WAM-risk stages versus v2 joint scene-trajectory flow, not simply RGB versus BEV.

## Contents

- `code/`: archived models, training, planning, configurations, scripts and tests. Shared base modules still live in the repository root; this directory is not a standalone runnable checkout.
- `docs/`: historical Stage 6/7 protocols and analyses.
- `artifacts/`: local-only `.tar.zst` archives. They are excluded from Git and have not been deleted.

Historical v1 scientific acceptance and v2 acceptance are distinct. Never compare different cohorts or treat a v1 gate as a v2 gate. v1 Stage 7 scientific acceptance failed. Selected evidence is published in [docs/evidence](../../docs/evidence/README.md).

For the complete old source layout use Git snapshot `58fedae458021bb37212442868288f95947288a8` in a separate worktree. Inspect archive contents before restoring local large files to a separate directory; do not extract over the active v2 source. See [reproduction instructions](../../docs/REPRODUCIBILITY.md).
