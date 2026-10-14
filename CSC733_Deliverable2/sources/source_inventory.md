# Phase 1 source inventory

Checked 2026-10-14 via upstream GitHub repository contents and documentation.

| Item | Source | What it offers | Notes |
|---|---|---|---|
| Main paper | https://doi.org/10.1145/3698816 | Claims, methods, figures and baselines | Academic source; cite |
| Repository root | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching | README, LICENSE, reproduction/, server/, wfpatch-extension/ | Authors label package SIGMOD 2025; paper published in 2024 |
| Reproduction guide | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/blob/main/reproduction/README.md | Configuration and scripts | States 130 hours for full reproduction |
| Redis experiment scripts | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/tree/main/reproduction/redis-cluster | Redis experiments and setup | Requires patched MMView kernel |
| Plotting scripts | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/tree/main/reproduction/plots | Figure recreation scripts | Transformed DuckDB data; R scripts |
| Original figures | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/tree/main/reproduction/original-plots | Visual baselines | Original images are not independent reproduction |
| Transformed data | https://doi.org/10.5281/zenodo.13729902 | Plottable DuckDB and non-DB inputs | Upstream README estimates 82 GB unpacked; not downloaded |
| Raw data | https://doi.org/10.5281/zenodo.13729902 | Raw benchmark inputs | Upstream README estimates 232 GB unpacked; not downloaded |
| VM | https://doi.org/10.5281/zenodo.13765240 | QEMU environment | Kernel support and virtualization limitations |
| WfPatch extension | https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/tree/main/wfpatch-extension | Library extension documentation | Reuse subject to original license |

## Exact figure-to-script mapping documented by authors
- Figure 1: `reproduction/plots/do-teaser-line`
- Figure 5: `reproduction/plots/do-synchronization-time-failover-line`
- Figure 6: `reproduction/plots/do-synchronization-time-patch-boxplot-idle`
- Figure 7: `reproduction/plots/do-all-patches`
- Figure 8: `reproduction/plots/do-rps-time`
- Figure 9: `reproduction/plots/do-rps-time-single-details`
- Figure 10: `reproduction/plots/do-latencies-single-details`

## Resource requirements quoted by original documentation
Authors recommend >=700 GB free space, >=384 GB RAM, >=22 CPU cores. Full reproducibility run estimated ~130 hours and ~685 GB generated data; original Redis experiments ~100 hours; data transformation / analysis ~6 hours. Redis live-patching experiments require a modified kernel; authors specifically say to use a regular kernel for DuckDB analysis.

## Phase 1 outstanding
- Pin full upstream git commit SHA using a verified current repository API commit response or local clone (prior Magnolia `git ls-remote HEAD` reported `58bc1b547ff129c4c7d48f41435ed52cd0dc520b`).
- Verify individual data archive contents and whether a manageable subset can be used without the full 82-GB extraction.
- Review original plotting scripts and dataset schema before promising analytic replication.
- No upstream source copied into this repo.
