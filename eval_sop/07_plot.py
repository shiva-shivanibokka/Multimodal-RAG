"""Plot risk-coverage curves from results/risk_coverage_curves.csv.
Needs only matplotlib + stdlib (the eval venv has no matplotlib; run with any python that has it)."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parent / "results"
curves = defaultdict(list)
with open(R / "risk_coverage_curves.csv") as f:
    for row in csv.DictReader(f):
        curves[row["signal"]].append((float(row["coverage"]), float(row["risk"])))
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, pts in curves.items():
    ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, lw=1.3)
ax.set_xlabel("coverage (fraction of questions answered)")
ax.set_ylabel("risk (1 - frac. answered that are in-corpus AND top-1 page correct)")
ax.set_title("Retrieval-gate risk-coverage (answerable-unambiguous + all OOD)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(R / "risk_coverage.png", dpi=130)
print("wrote", R / "risk_coverage.png")
