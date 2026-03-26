"""
FL Security Testbed — CLI experiment runner.

Entry point for the ``fl-run`` console script defined in pyproject.toml.

Usage:
    fl-run --help
    fl-run run --attack label_flip --defense krum --rounds 10
    fl-run list-attacks
    fl-run list-defenses
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:  # pragma: no cover
    print("typer not installed — run: pip install typer", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fl_runner")

app = typer.Typer(
    name="fl-run",
    help="FL Security Testbed experiment runner.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    attack: str = typer.Option("none", help="Attack strategy (label_flip, min_max, min_sum, none)."),
    defense: str = typer.Option("fedavg", help="Aggregation defense (fedavg, krum, trimmed_mean, median, flame)."),
    rounds: int = typer.Option(5, help="Number of federated rounds."),
    n_clients: int = typer.Option(10, help="Total number of clients."),
    byzantine_fraction: float = typer.Option(0.2, help="Fraction of Byzantine clients [0, 0.5)."),
    seed: int = typer.Option(42, help="Random seed."),
    output: Optional[Path] = typer.Option(None, help="Write JSON results to this path."),
) -> None:
    """Run a federated learning experiment with configurable attack and defense."""
    from server.aggregation import get_aggregation_fn

    n_byzantine = max(0, int(n_clients * byzantine_fraction))
    n_honest = n_clients - n_byzantine

    log.info(
        "Experiment: attack=%s  defense=%s  rounds=%d  clients=%d (%d honest, %d byzantine)",
        attack, defense, rounds, n_clients, n_honest, n_byzantine,
    )

    try:
        agg_fn = get_aggregation_fn(defense)
    except ValueError as exc:
        typer.echo(f"Unknown defense: {exc}", err=True)
        raise typer.Exit(code=1)

    import numpy as np
    rng = np.random.default_rng(seed)

    results: list[dict] = []
    for rnd in range(1, rounds + 1):
        # Simulate honest updates (unit vectors near 1.0)
        honest_grads = [[rng.standard_normal((16,)).astype("float32")] for _ in range(n_honest)]

        # Simulate Byzantine updates
        if attack == "label_flip":
            byzantine_grads = [[-g[0] * 3.0] for g in honest_grads[:n_byzantine]]
        elif attack in ("min_max", "min_sum"):
            byzantine_grads = [[rng.standard_normal((16,)).astype("float32") * 50.0]
                               for _ in range(n_byzantine)]
        else:
            byzantine_grads = [[rng.standard_normal((16,)).astype("float32")]
                               for _ in range(n_byzantine)]

        all_grads = honest_grads + byzantine_grads
        agg = agg_fn(all_grads)
        norm = float(sum(float(g.flatten().__abs__().mean()) for g in agg))

        log.info("Round %d/%d  agg_norm=%.4f", rnd, rounds, norm)
        results.append({"round": rnd, "agg_norm": norm})

    summary = {
        "attack": attack,
        "defense": defense,
        "rounds": rounds,
        "n_clients": n_clients,
        "byzantine_fraction": byzantine_fraction,
        "results": results,
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        log.info("Results written to %s", output)
    else:
        typer.echo(json.dumps(summary, indent=2))


@app.command(name="list-attacks")
def list_attacks() -> None:
    """List available attack strategies."""
    attacks = ["none", "label_flip", "min_max", "min_sum"]
    for a in attacks:
        typer.echo(f"  {a}")


@app.command(name="list-defenses")
def list_defenses() -> None:
    """List available defense / aggregation strategies."""
    defenses = ["fedavg", "krum", "multi_krum", "trimmed_mean", "median", "flame"]
    for d in defenses:
        typer.echo(f"  {d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
