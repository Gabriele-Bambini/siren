#!/usr/bin/env python3
"""Fig 1 v2 — The Specificity Wall (publication styling)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.8, "savefig.facecolor": "white"})

RES = os.path.expanduser("~/siren_hpc/results")
M = {
    "K562 · chr19 test":         {"GBM dist+ATAC (K562)": 0.1212, "GBM trained on Gasperini": 0.004},
    "K562 · held-out chroms":    {"GBM dist+ATAC (K562)": 0.1823, "GBM trained on Gasperini": None},
    "HCT116 · held-out screen":  {"GBM dist+ATAC (K562)": 0.0143, "GBM trained on Gasperini": None},
    "WTC11 · held-out screen":   {"GBM dist+ATAC (K562)": 0.0398, "GBM trained on Gasperini": None},
}
rows = ["GBM dist+ATAC (K562)", "GBM trained on Gasperini"]
cols = list(M.keys())
Z = np.array([[np.nan if M[c].get(r) is None else M[c][r] for c in cols] for r in rows])

fig, ax = plt.subplots(figsize=(9.2, 3.6))
im = ax.imshow(Z, cmap="RdYlGn", vmin=-0.05, vmax=0.25, aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10, fontweight="bold")
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="both", length=0)
for i in range(len(rows)):
    for j in range(len(cols)):
        v = Z[i, j]
        if np.isnan(v):
            ax.text(j, i, "n.t.", ha="center", va="center", color="#888888",
                    fontsize=10, style="italic"); continue
        strong = v >= 0.12
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if strong else ("#7a0000" if v < 0.05 else "#222222"),
                fontsize=12, fontweight="bold" if strong else "normal")
ax.set_title("The specificity wall", fontsize=13, fontweight="bold", pad=6, loc="left")
ax.text(0, 1.06, "Spearman ρ on |CRISPRi effect| — signal survives only within the training cell type and screen",
        transform=ax.transAxes, fontsize=10, color="#444444")
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
cb.set_label("ρ (magnitude)", fontsize=9); cb.ax.tick_params(labelsize=8)
note = ("Within-cell-type generalization is strong (ρ = 0.18 on unseen chromosomes); cross-cell-type and "
        "cross-screen transfer collapse.\nEnformer ISD deltas: Δρ −0.275 [−0.21, +0.13] (powered, pre-registered). "
        "Cross-screen ensemble underperforms the native model (0.058 < 0.106).")
fig.text(0.005, -0.02, note, fontsize=8.5, ha="left", color="#333333", style="italic")
fig.tight_layout()
fig.savefig(f"{RES}/fig_transfer_matrix.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RES}/fig_transfer_matrix.pdf", bbox_inches="tight")
json.dump({"matrix": {r: {c: M[c][r] for c in cols} for r in rows}},
          open(f"{RES}/fig_transfer_matrix.json", "w"), indent=2)
print("FIG_V2_DONE")
