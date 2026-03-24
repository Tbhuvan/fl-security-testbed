<div align="center">

# FL-Security-Testbed

**Federated learning security research environment with Byzantine attack and defence simulation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## Overview

FL-Security-Testbed is a research environment for studying security properties of federated learning systems. It provides implementations of Byzantine attacks and robust aggregation defences, enabling controlled experiments on how malicious participants can degrade federated model quality — and how to prevent it.

**Research question:** Can poisoned participants in a federated vulnerability probe training system degrade the collective probe's accuracy? This testbed provides the infrastructure to answer that question.

## Implemented Attacks

| Attack | Description | Reference |
|--------|-------------|-----------|
| Byzantine Gradient | Arbitrary gradient manipulation | Blanchard et al., 2017 |
| Label Flip | Systematic label corruption | Bagdasaryan et al., 2018 |
| Min-Max | Maximise damage while evading detection | Shejwalkar & Houmansadr, 2021 |
| Min-Sum | Minimise aggregate perturbation norm | Shejwalkar & Houmansadr, 2021 |

## Implemented Defences

| Defence | Description | Reference |
|---------|-------------|-----------|
| FedAvg | Standard averaging (baseline, no defence) | McMahan et al., 2017 |
| Krum | Distance-based outlier rejection | Blanchard et al., 2017 |
| Trimmed Mean | Coordinate-wise trimmed aggregation | Yin et al., 2018 |
| Median | Coordinate-wise median aggregation | Yin et al., 2018 |
| FLAME | Clustering-based Byzantine filtering | Nguyen et al., 2022 |

## Quick Start

```bash
pip install -e .

# Run a federated training experiment
python run.py --attack byzantine --defence krum --rounds 50 --malicious 0.3

# Run full experiment suite
python experiments/runner.py --config experiments/config.yaml
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
├── fl_testbed/      # Core importable library (attacks, defenses, data, models)
├── clients/         # Federated client implementations + model + data loading
├── server/          # Flower server and aggregation strategies
├── attacks/         # Byzantine attack implementations + attack factory
├── defenses/        # Defence strategy implementations
├── experiments/     # Experiment configs and runner
├── tests/           # Test suite
└── run.py           # Entry point
```

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Key Papers

- FedAvg: McMahan et al. 2017 -- arXiv:1602.05629
- Krum: Blanchard et al. 2017 -- arXiv:1703.02757
- Trimmed Mean / Median: Yin et al. 2018 -- arXiv:1803.01498
- FLAME: Nguyen et al. 2022 -- arXiv:2101.02281
- Backdoor FL: Bagdasaryan et al. 2018 -- arXiv:1807.00459
- Byzantine ML survey: Lyu et al. 2020 -- arXiv:2003.02445
- Differential Privacy FL: McMahan et al. 2018 -- arXiv:1710.06963

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. This testbed provides the experimental platform for RQ5: training vulnerability probes collaboratively across organisations without exposing proprietary code, while defending against poisoning attacks on the shared probe.

## License

Apache License 2.0
