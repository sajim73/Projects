# CSC 733 Group 2 — Live Patching Reproducibility Project

**Status:** Phase 1 — source preparation and evidence audit. No original WfPatch experiments or new simulations have been run by this project yet.

## Scope
Original paper: Michael Fruth and Stefanie Scherzinger, *Live Patching for Distributed In-Memory Key-Value Stores*, Proc. ACM Manag. Data 2(6), article 241 (2024). DOI: https://doi.org/10.1145/3698816

Research extensions:
- RQ1: effects of network delay, jitter and packet loss on patch distribution;
- RQ2: effects of temporary node disconnections on patch convergence and recovery;
- RQ3: cumulative resource and performance costs of 0–100 successive patches.

## Original sources
- Reproduction repo: https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching
- Data: https://doi.org/10.5281/zenodo.13729902
- Preconfigured QEMU VM: https://doi.org/10.5281/zenodo.13765240
- Reproduction instructions: https://github.com/sdbs-uni-p/sigmod25-distributed-imkvs-live-patching/blob/main/reproduction/README.md
- SIGMOD ARI package requirements: https://docs.google.com/document/d/1_pheZ2p9Nc8qhtcOpNINm7AxFpPpkpC1n60jJdyr-uk/export?format=pdf&attachment=false

Do not mistake this directory for a clone or fork of the authors' repository. It is our source inventory and planned extension workspace. Preserve original licenses and attribution if copying code.

## Phase 1 evidence
See [sources/source_inventory.md](sources/source_inventory.md), [sources/evidence_inventory.csv](sources/evidence_inventory.csv), and [sources/provenance.md](sources/provenance.md).

## Remaining work
1. Check availability of selective transformed source data and pin an upstream commit.
2. Implement actual analytic/model experiments and tests for RQ1–RQ3.
3. Generate raw CSV data, analyses, and figures from executable code.
4. Package a clean runnable artifact (Docker if feasible) and verify end-to-end.
5. Write Section 3 and Section 5 accurately. Source-derived and simulated findings must be identified as such.

**Important:** Original implementation needs a modified MMView kernel. We cannot claim independent runtime reproducibility from source inspection or original plots alone.
