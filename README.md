<div align="center">

# FL-Security-Testbed

**Federated learning security research environment — Byzantine attacks, robust aggregation, and poisoning defences**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## Overview

FL-Security-Testbed is a research environment for studying security properties of federated learning systems under Byzantine attack. It provides implementations of four attack strategies and five aggregation defences, enabling controlled experiments on how malicious participants degrade federated model quality — and how robust aggregation prevents it.

**Research question:** Can poisoned participants in a federated vulnerability probe training system degrade the collective probe's accuracy below a useful threshold? This testbed provides the infrastructure and benchmarks to answer that question.

## Aggregation Robustness Benchmark

Robustness score = 1 − ||aggregated − honest mean|| / ||honest mean||. Score of 1.0 = perfect recovery; score < 0 = the attack actively pushes the aggregate away from the honest mean (active divergence).

**At 20% Byzantine fraction (240 trials, 3 seeds, gradient dim=50):**

| Defence | No Attack | Random Noise | Sign Flip | Gradient Scale |
|---------|-----------|--------------|-----------|----------------|
| **FedAvg** (baseline) | 1.000 | −5.92 | −0.20 | **−8.80** |
| **Krum** | 1.000 | 0.889 | 0.889 | 0.889 |
| **Trimmed Mean** | 0.990 | 0.972 | 0.943 | 0.943 |
| **Coordinate Median** | 0.996 | 0.968 | 0.949 | 0.950 |
| **FLAME** | 0.989 | 0.862 | **0.995** | **0.991** |

**At 40% Byzantine fraction:**

| Defence | Random Noise | Sign Flip | Gradient Scale |
|---------|--------------|-----------|----------------|
| FedAvg | −9.11 | −1.40 | −18.60 |
| Krum | 0.889 | 0.889 | 0.889 |
| Trimmed Mean | 0.950 | 0.855 | 0.855 |
| Coordinate Median | 0.950 | 0.862 | 0.864 |
| FLAME | 0.743 | 0.999 | 0.986 |

**Key findings:**
- FedAvg collapses to robustness −18.6 under gradient scaling at 40% Byzantine fraction, confirming theoretical results from Blanchard et al. (2017)
- FLAME achieves near-perfect robustness (>0.99) on direction-preserving attacks (sign flip, gradient scale) via norm-clipping; random noise is harder because Byzantine gradients can contaminate the dominant cosine cluster
- Trimmed Mean and Coordinate Median offer the best overall balance — robustness >0.85 across all attacks at 40% Byzantine fraction with no hyperparameter tuning
- Krum's robustness ceiling at ~0.889 reflects the irreducible variance of selecting a single gradient rather than averaging all honest clients

*Full results: `experiments/results/aggregation_benchmark.json` · Heatmap: `experiments/results/aggregation_heatmap.png`*

## Implemented Attacks

| Attack | Description | Reference |
|--------|-------------|-----------|
| Random Noise | Byzantine clients send large-magnitude random gradients | Blanchard et al., 2017 |
| Sign Flip | Send negated honest gradient scaled ×5 | Bhagoji et al., 2019 |
| Gradient Scale | Send honest gradient amplified ×50 | Shejwalkar & Houmansadr, 2021 |
| Label Flip | Systematic label corruption during training | Bagdasaryan et al., 2018 |

## Implemented Defences

| Defence | Description | Reference |
|---------|-------------|-----------|
| FedAvg | Standard averaging — no defence (baseline) | McMahan et al., 2017 |
| Krum | Select the gradient closest to its k-nearest neighbours | Blanchard et al., 2017 |
| Trimmed Mean | Coordinate-wise trimmed aggregation | Yin et al., 2018 |
| Coordinate Median | Coordinate-wise median aggregation | Yin et al., 2018 |
| FLAME | Norm-clipping + cosine clustering + DP noise | Nguyen et al., 2022 |

## Quick Start

```bash
pip install -e .

# Run the aggregation robustness benchmark (pure numpy, no GPU required)
python experiments/aggregation_benchmark.py

# Run a federated training experiment via CLI
fl-run run --attack sign_flip --defense krum --rounds 10 --byzantine-fraction 0.2

# List available attacks and defences
fl-run list-attacks
fl-run list-defenses
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client 1   │     │   Client 2   │     │  Client 3    │
│   (honest)   │     │  (malicious) │     │   (honest)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                     │
       └────────────┬───────┴──────────┬──────────┘
                    │                  │
              ┌─────▼──────────────────▼─────┐
              │     Aggregation Server       │
              │  (Krum / TrimmedMean / etc.) │
              └──────────────────────────────┘
```

Built on [Flower (flwr)](https://flower.dev/) for production-grade federated learning orchestration.

## Project Structure

```
fl-security-testbed/
├── fl_testbed/          # Core importable library + CLI runner (fl-run entry point)
├── server/              # Aggregation strategies: FedAvg, Krum, TrimmedMean, Median, FLAME
├── attacks/             # Byzantine attack implementations + attack registry
├── clients/             # FL client, local training, dataset partitioning (IID + non-IID)
├── experiments/
│   ├── aggregation_benchmark.py   # Full attack × defence robustness matrix (pure numpy)
│   ├── results/
│   │   ├── aggregation_benchmark.json  # 240-trial results
│   │   └── aggregation_heatmap.png     # Publication-ready heatmap
│   └── runner.py        # Flower-based full training experiment runner
├── tests/               # 49 tests covering aggregation, attacks, models
└── run.py               # Flower simulation entry point
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Papers

- FedAvg: McMahan et al. 2017 — arXiv:1602.05629
- Krum: Blanchard et al. 2017 — arXiv:1703.02757
- Trimmed Mean / Median: Yin et al. 2018 — arXiv:1803.01498
- FLAME: Nguyen et al. 2022 — arXiv:2101.02281
- Backdoor FL: Bagdasaryan et al. 2018 — arXiv:1807.00459
- Min-Max / Min-Sum: Shejwalkar & Houmansadr 2021 — arXiv:2103.06257

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. The open question this testbed targets: can vulnerability detection probes be trained collaboratively across organisations without exposing proprietary code — and what happens when one participant poisons the shared model?

## License

Apache License 2.0
