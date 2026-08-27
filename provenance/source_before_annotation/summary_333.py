# summary_333.py
# Report figure and table for Section 3.3.3, generated from
# heavy_333_records.csv plus the matching Gaussian cells of
# allocation_332_records.csv (G4-N at (48,192) and (216,24)), which are
# reused, not rerun. Writes table_333.tex and figure_333.pdf / .png.
# Run in a folder containing both CSV files:  python summary_333.py

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALLOCS = [("(48,192)", r"$\rho=0.25$ (48,192), strong help", 0, 0),
          ("(216,24)", r"$\rho=9$ (216,24), near no help", 1, 4)]
DISTS = ["Gaussian", "t5", "t3"]


def group_stats(ratios, zero, sent, fails):
    return {"ratio": ratios,
            "median": np.median(ratios),
            "q25": np.quantile(ratios, .25),
            "q75": np.quantile(ratios, .75),
            "q05": np.quantile(ratios, .05),
            "q95": np.quantile(ratios, .95),
            "improve": np.mean(ratios < 1.0),
            "zero": zero, "sentinel": sent, "failures": fails}


# heavy-tail cells (compute)
heavy = list(csv.DictReader(open("heavy_333_records.csv")))
gauss = list(csv.DictReader(open("allocation_332_records.csv")))
groups = {}
for label, _, ai_h, ai_g in ALLOCS:
    sub = [r for r in gauss if r["geometry"] == "G4-N"
           and int(r["alloc_index"]) == ai_g and r["failure"] == ""]
    nfail = sum(1 for r in gauss if r["geometry"] == "G4-N"
                and int(r["alloc_index"]) == ai_g and r["failure"] != "")
    ratios = np.array([float(r["L_RBPB"]) / float(r["L0"]) for r in sub])
    groups[("Gaussian", label)] = group_stats(
        ratios, np.mean([r["zero"] == "True" for r in sub]),
        np.mean([r["sentinel"] == "True" for r in sub]), nfail)
    for dist in ("t5", "t3"):
        sub = [r for r in heavy if r["distribution"] == dist
               and int(r["alloc_index"]) == ai_h and r["failure"] == ""]
        nfail = sum(1 for r in heavy if r["distribution"] == dist
                    and int(r["alloc_index"]) == ai_h and r["failure"] != "")
        ratios = np.array([float(r["L_RBPB"]) / float(r["L0"])
                           for r in sub])
        groups[(dist, label)] = group_stats(
            ratios, np.mean([r["zero"] == "True" for r in sub]),
            np.mean([r["sentinel"] == "True" for r in sub]), nfail)

# figure: one panel per allocation, median + IQR + 90% interval (display)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (label, title, _, _) in zip(axes, ALLOCS):
    for x, dist in enumerate(DISTS):
        g = groups[(dist, label)]
        ax.vlines(x, g["q05"], g["q95"], color="C0", lw=1.2, alpha=0.8)
        ax.vlines(x, g["q25"], g["q75"], color="C0", lw=7, alpha=0.45)
        ax.plot(x, g["median"], "o", color="C1", ms=6, zorder=5)
        ax.annotate(f"{g['median']:.3f}", (x, g["median"]),
                    textcoords="offset points", xytext=(10, -3),
                    fontsize=8, color="C1")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Gaussian\n(from 3.3.2)", r"$t_5$", r"$t_3$"])
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, 2.5)
axes[0].set_ylabel(r"paired loss ratio $L_{\mathrm{RBPB}} / L_{0}$")
fig.tight_layout()
fig.savefig("figure_333.pdf", bbox_inches="tight")
fig.savefig("figure_333.png", dpi=150, bbox_inches="tight")

# compact table per G7 (record)
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"Distribution & $(n_1,n_2)$ & median & IQR & "
    r"$\Pr(L_{\mathrm{RBPB}}<L_0)$ & zero rate & "
    r"sentinel rate / failures \\",
    r"\midrule",
]
for label, _, _, _ in ALLOCS:
    for dist in DISTS:
        g = groups[(dist, label)]
        name = {"Gaussian": "Gaussian", "t5": "$t_5$", "t3": "$t_3$"}[dist]
        lines.append(
            f"{name} & {label} & {g['median']:.4f} & "
            f"[{g['q25']:.4f}, {g['q75']:.4f}] & {g['improve']:.3f} & "
            f"{g['zero']:.3f} & "
            f"{g['sentinel']:.4f} / {g['failures']} \\\\")
    if label != ALLOCS[-1][0]:
        lines.append(r"\midrule")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Sensitivity of the frozen Gaussian RBPB selector to "
    r"heavy-tailed sampling with an exactly known transport map: paired "
    r"loss ratios against the unregularised map, with improvement "
    r"probability, zero-selection and sentinel-selection rates, and "
    r"failure counts. Gaussian rows reuse the matching 3.3.2 "
    r"confirmatory records. The selector, grid and budget are unchanged "
    r"from the Gaussian sections. The reported $t_3$ summaries are "
    r"quantile-based because $t_3$ lacks a finite fourth moment, so "
    r"ordinary variance-based Monte Carlo standard errors are not used.}",
    r"\label{tab:heavy-333}",
    r"\end{table}",
]
open("table_333.tex", "w").write("\n".join(lines) + "\n")

print(f"{'dist':9s} {'alloc':9s} {'median':>7s} {'IQR':>17s} "
      f"{'improve':>8s} {'zero':>6s} {'sent':>6s}")
for label, _, _, _ in ALLOCS:
    for dist in DISTS:
        g = groups[(dist, label)]
        print(f"{dist:9s} {label:9s} {g['median']:7.4f} "
              f"[{g['q25']:.4f}, {g['q75']:.4f}] {g['improve']:8.3f} "
              f"{g['zero']:6.3f} {g['sentinel']:6.3f}")
print("\nwritten: table_333.tex, figure_333.pdf, figure_333.png")