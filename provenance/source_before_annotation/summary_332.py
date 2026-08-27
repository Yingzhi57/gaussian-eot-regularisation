# summary_332.py
# Report table and main figure for Section 3.3.2, generated purely from
# the existing run outputs (allocation_332_records.csv and
# allocation_332_reference.npz); nothing is re-simulated.
# Writes: table_332.tex (booktabs + multirow), figure_332.pdf / .png,
# and prints a plain-text preview of the table.
# The vertical line at rho = 11 is the known-basis leading reference and
# the caption must say so; it is not a full-covariance threshold.
# Run: python summary_332.py     (in the folder with the two data files)

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GEOMS = ["G4-C", "G4-N", "G4-K"]
ETAS = {"G4-C": 0.0, "G4-N": 0.133522, "G4-K": 0.295693}
DESCS = {"G4-C": "commuting control",
         "G4-N": "moderate non-commuting",
         "G4-K": "conditioning stress"}
RHOS = [0.25, 1.0, 3.0, 7.0, 9.0, 11.0, 15.0]


def ratio_and_se(Lv, L0v):
    # paired delta-method SE for the risk ratio mean(Lv)/mean(L0v)
    rho = Lv.mean() / L0v.mean()
    infl = Lv - rho * L0v
    return rho, infl.std(ddof=1) / np.sqrt(len(L0v)) / L0v.mean()


# aggregate the confirmatory records per cell (compute)
rows = [r for r in csv.DictReader(open("allocation_332_records.csv"))
        if r["failure"] == ""]
z = np.load("allocation_332_reference.npz", allow_pickle=True)

cells = {}
for g in GEOMS:
    for ai, rho in enumerate(RHOS):
        sub = [r for r in rows
               if r["geometry"] == g and int(r["alloc_index"]) == ai]
        L0 = np.array([float(r["L0"]) for r in sub])
        stats = {}
        for key, col in (("KB", "L_KB"), ("dense", "L_dense"),
                         ("RBPB", "L_RBPB")):
            stats[key] = ratio_and_se(
                np.array([float(r[col]) for r in sub]), L0)
        cells[(g, ai)] = {
            "rho": rho,
            "m_kb": max(0.0, (11 - rho) / (1 + rho)),
            "m_dense": float(z[f"{g}_{ai}_m_dense"]),
            "stats": stats,
            "zero": np.mean([r["zero"] == "True" for r in sub]),
            "m_hat": np.array([float(r["m_rbpb"]) for r in sub]),
            "n": len(sub),
        }

# LaTeX table (needs \usepackage{booktabs, multirow}) (record)
max_se = max(se for c in cells.values() for _, se in c["stats"].values())
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\begin{tabular}{llrrrrrr}",
    r"\toprule",
    r"Geometry & $\rho=n_1/n_2$ & $m_{\mathrm{KB}}^{(1)}(\rho)$ & $\widehat m_{\mathrm{dense}}$ & "
    r"$R_{\mathrm{KB}}/R_{0}$ & $R_{\mathrm{dense}}/R_{0}$ & "
    r"$R_{\mathrm{RBPB}}/R_{0}$ & RBPB zero-selection rate \\",
    r"\midrule",
]
for g in GEOMS:
    for ai, rho in enumerate(RHOS):
        c = cells[(g, ai)]
        head = (rf"\multirow{{7}}{{*}}{{{g}}}" if ai == 0 else "")
        lines.append(
            f"{head} & {rho:g} & {c['m_kb']:g} & {c['m_dense']:g} & "
            f"{c['stats']['KB'][0]:.4f} & {c['stats']['dense'][0]:.4f} & "
            f"{c['stats']['RBPB'][0]:.4f} & {c['zero']:.3f} \\\\")
    if g != GEOMS[-1]:
        lines.append(r"\midrule")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Population-risk ratios of the fixed known-basis reference "
    r"$m_{\mathrm{KB}}^{(1)}(\rho)$, the independently estimated best common multiplier \widehat m_{\mathrm{dense}}"
    r"$\widehat m_{\mathrm{dense}}$ and the RBPB selection, relative to the "
    r"unregularised map, with the RBPB zero-selection rate; "
    rf"paired delta-method MCSEs are at most {max_se:.4f}. "
    r"At $\rho=11$, the displayed known-basis leading comparator is zero; "
    r"this is not a full-matrix threshold.}",
    r"\label{tab:allocation-332}",
    r"\end{table}",
]
open("table_332.tex", "w").write("\n".join(lines) + "\n")

# plain-text preview (display)
print(f"{'geom':6s} {'rho':>5s} {'mKB':>5s} {'m*':>7s} "
      f"{'KB':>7s} {'dense':>7s} {'RBPB':>7s} {'zero':>6s} {'n':>6s}")
for g in GEOMS:
    for ai, rho in enumerate(RHOS):
        c = cells[(g, ai)]
        print(f"{g:6s} {rho:5g} {c['m_kb']:5g} {c['m_dense']:7g} "
              f"{c['stats']['KB'][0]:7.4f} {c['stats']['dense'][0]:7.4f} "
              f"{c['stats']['RBPB'][0]:7.4f} {c['zero']:6.3f} {c['n']:6d}")

# main figure: gains on the left, selector behaviour on the right
fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.6), sharex=True)
for gi, g in enumerate(GEOMS):
    rhos = np.array(RHOS)
    gain = {k: 100 * (1 - np.array([cells[(g, a)]["stats"][k][0]
                                    for a in range(7)]))
            for k in ("KB", "dense", "RBPB")}
    se_rb = 100 * np.array([cells[(g, a)]["stats"]["RBPB"][1]
                            for a in range(7)])
    m_star = np.array([cells[(g, a)]["m_dense"] for a in range(7)])
    m_kb = np.array([cells[(g, a)]["m_kb"] for a in range(7)])
    zero = np.array([cells[(g, a)]["zero"] for a in range(7)])
    med = np.array([np.median(cells[(g, a)]["m_hat"]) for a in range(7)])
    q25 = np.array([np.quantile(cells[(g, a)]["m_hat"], .25)
                    for a in range(7)])
    q75 = np.array([np.quantile(cells[(g, a)]["m_hat"], .75)
                    for a in range(7)])

    pL, pR = axes[gi]
    pL.plot(rhos, gain["dense"], ls=(0, (5, 3)), color="C3",
            lw=3.2, alpha=0.9,
            label=r"$\widehat m_{\mathrm{dense}}$")
    pL.plot(rhos, gain["KB"], ":", color="C2", alpha=0.9,
            label=r"$m_{\mathrm{KB}}^{(1)}(\rho)$")
    pL.errorbar(rhos, gain["RBPB"], yerr=se_rb, fmt="o-", color="C1",
                ms=4, capsize=2, alpha=1., label="RBPB")
    pL.axvline(11, color="grey", ls="-", lw=0.8, alpha=0.5)
    pL.axhline(0, color="k", lw=0.6)
    pL.set_xscale("log")
    pL.set_ylabel("Pop-risk gain relative to $m=0$ (%)")
    pL.set_title(f"{DESCS[g]}  "
                 rf"($\kappa={int(np.linalg.cond(z[f'{g}_Sigma1']))}$, "
                 rf"$\eta={ETAS[g]:.3f}$)",
                 fontsize=10,
                 loc="left",
                 )

    pR.plot(
        rhos, m_star, "s-", color="C3", ms=4, alpha=0.85,
        label=r"$\widehat m_{\mathrm{dense}}$",
    )
    pR.plot(
        rhos, m_kb, ":", color="C2", alpha=0.9,
        label=r"$m_{\mathrm{KB}}^{(1)}(\rho)$",
    )
    pR.errorbar(
        rhos, med, 
        yerr=[med - q25, q75 - med], 
        fmt="o-", color="C1", ms=4, capsize=2, alpha=0.85,
        label=r"RBPB $\widehat m_B$ (median, IQR)",
        )
    for x, m, lo, hi in zip(rhos, med, q25, q75):
        if m < 3:      # label above the upper IQR whisker
            pR.annotate(f"{m:g}", (x, hi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color="C1")
        else:          # label below the lower IQR whisker
            pR.annotate(f"{m:g}", (x, lo), textcoords="offset points",
                        xytext=(0, -9), ha="center", fontsize=7, color="C1")
    pR.axvline(11, color="grey", ls="-", lw=0.8, alpha=0.5)
    pR.set_ylabel("multiplier")
    pRz = pR.twinx()
    pRz.plot(rhos, zero, "^-", color="C0", ms=4, alpha=0.7,
             label="RBPB zero-selection rate")
    pRz.set_ylim(-0.03, 1.03)
    pRz.set_ylabel("zero-selection rate")
    if gi == 0:
        pL.legend(fontsize=8, loc="upper right")
        h1, l1 = pR.get_legend_handles_labels()
        h2, l2 = pRz.get_legend_handles_labels()
        axes[0, 1].legend(h1 + h2, l1 + l2, fontsize=8, ncol=2,
                          loc="lower center", bbox_to_anchor=(0.5, 1.03))
for ax in axes[-1]:
    ax.set_xlabel(r"$\rho = n_1/n_2$ (log scale)")
    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"{r:g}" for r in RHOS])
fig.tight_layout()
fig.savefig("figure_332.pdf", bbox_inches="tight")
fig.savefig("figure_332.png", dpi=150, bbox_inches="tight")
print("\nwritten: table_332.tex, figure_332.pdf, figure_332.png")