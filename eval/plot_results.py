"""Graphiques à partir de eval/resultats/experiences.csv.

docker compose exec api python -m eval.plot_results
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.run_eval import RESULTS_DIR, SUMMARY_FILE  # noqa: E402


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    if not SUMMARY_FILE.exists():
        raise SystemExit("Aucun résultat : lance d'abord python -m eval.run_eval")
    with SUMMARY_FILE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    # Dernière exécution de chaque expérience (tag)
    latest = {row["tag"]: row for row in rows}
    tags = list(latest)

    charts = [
        ("hit_rate", "Hit Rate@k", "recherche_hit_rate.png"),
        ("mrr", "MRR", "recherche_mrr.png"),
        ("latence_mediane_s", "Latence médiane (s)", "latence.png"),
        ("pertinence_0_2", "Pertinence moyenne (0-2)", "pertinence.png"),
    ]
    for key, label, filename in charts:
        points = [(t, _float(latest[t].get(key))) for t in tags]
        points = [(t, v) for t, v in points if v is not None]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar([t for t, _ in points], [v for _, v in points], color="#2b3595")
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylabel(label)
        ax.set_title(f"{label} par expérience")
        if key in {"hit_rate", "mrr"}:
            ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=120)
        plt.close(fig)
        print(f"Graphique : {RESULTS_DIR / filename}")


if __name__ == "__main__":
    main()
