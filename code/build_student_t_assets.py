"""Build the supplementary Student-t figure and table from saved records."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_RECORDS = REPO_ROOT / "data/student_t/heavy_333_records.csv"
GAUSSIAN_RECORDS = REPO_ROOT / "data/allocation/allocation_332_records.csv"
FIGURE_DIR = REPO_ROOT / "results/figures"
TABLE_PATH = REPO_ROOT / "results/tables/table_student_t_sensitivity.tex"

ALLOCATIONS = (
    ("(48,192)", r"$\rho=0.25$ (48,192), target-rich allocation", 0, 0),
    ("(216,24)", r"$\rho=9$ (216,24), source-heavy allocation", 1, 4),
)
SAMPLING_LAWS = ("Gaussian", "t5", "t3")


def group_stats(ratios, zero_rate):
    """Return the paired-ratio and zero-selection summaries used in the report."""
    return {
        "median": np.median(ratios),
        "q25": np.quantile(ratios, 0.25),
        "q75": np.quantile(ratios, 0.75),
        "q05": np.quantile(ratios, 0.05),
        "q95": np.quantile(ratios, 0.95),
        "improve": np.mean(ratios < 1.0),
        "zero": zero_rate,
    }


def load_groups():
    """Combine Student-t records with the matching Gaussian evaluations."""
    with HEAVY_RECORDS.open(newline="") as stream:
        heavy = list(csv.DictReader(stream))
    with GAUSSIAN_RECORDS.open(newline="") as stream:
        gaussian = list(csv.DictReader(stream))

    groups = {}
    for label, _, heavy_index, gaussian_index in ALLOCATIONS:
        subset = [
            row
            for row in gaussian
            if row["geometry"] == "G4-N"
            and int(row["alloc_index"]) == gaussian_index
            and row["failure"] == ""
        ]
        ratios = np.array(
            [float(row["L_RBPB"]) / float(row["L0"]) for row in subset]
        )
        groups[("Gaussian", label)] = group_stats(
            ratios, np.mean([row["zero"] == "True" for row in subset])
        )

        for law in ("t5", "t3"):
            subset = [
                row
                for row in heavy
                if row["distribution"] == law
                and int(row["alloc_index"]) == heavy_index
                and row["failure"] == ""
            ]
            ratios = np.array(
                [float(row["L_RBPB"]) / float(row["L0"]) for row in subset]
            )
            groups[(law, label)] = group_stats(
                ratios, np.mean([row["zero"] == "True" for row in subset])
            )
    return groups


def build_figure(groups):
    """Plot paired loss-ratio quantiles for each allocation."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (label, title, _, _) in zip(axes, ALLOCATIONS):
        for x_value, law in enumerate(SAMPLING_LAWS):
            stats = groups[(law, label)]
            axis.vlines(
                x_value, stats["q05"], stats["q95"], color="C0", lw=1.2,
                alpha=0.8
            )
            axis.vlines(
                x_value, stats["q25"], stats["q75"], color="C0", lw=7,
                alpha=0.45
            )
            axis.plot(x_value, stats["median"], "o", color="C1", ms=6, zorder=5)
            axis.annotate(
                f"{stats['median']:.3f}", (x_value, stats["median"]),
                textcoords="offset points", xytext=(10, -3), fontsize=8,
                color="C1"
            )
        axis.axhline(1.0, color="k", ls=":", lw=1)
        axis.set_xticks(range(3))
        axis.set_xticklabels(["Gaussian\n(main evaluation)", r"$t_5$", r"$t_3$"])
        axis.set_title(title, fontsize=10)
        axis.set_xlim(-0.5, 2.5)
    axes[0].set_ylabel(r"paired loss ratio $L_{\mathrm{RBPB}} / L_{0}$")
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / "figure_student_t_sensitivity.pdf", bbox_inches="tight"
    )
    figure.savefig(
        FIGURE_DIR / "figure_student_t_sensitivity.png", dpi=150,
        bbox_inches="tight"
    )


def build_table(groups):
    """Write the paired loss and selection summaries used in the supplement."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Sampling law & $(n_1,n_2)$ & \shortstack{Median ratio\\IQR} & "
        r"$\Pr(L_{\mathrm{RBPB}}<L_0)$ & Zero-selection rate \\",
        r"\midrule",
    ]
    for label, _, _, _ in ALLOCATIONS:
        for law in SAMPLING_LAWS:
            stats = groups[(law, label)]
            name = {"Gaussian": "Gaussian", "t5": "$t_5$", "t3": "$t_3$"}[law]
            lines.append(
                f"{name} & ${label}$ & "
                f"${stats['median']:.4f}\\;[{stats['q25']:.4f},"
                f"{stats['q75']:.4f}]$ & ${stats['improve']:.3f}$ & "
                f"${stats['zero']:.3f}$ \\\\"
            )
        if label != ALLOCATIONS[-1][0]:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Paired loss and selection summaries for the Student-$t$ "
        r"sensitivity study and matching Gaussian cells. The ratio compares the "
        r"selected RBPB map with the unregularised plug-in on the same dataset. "
        r"Quantile summaries are reported rather than variance-based Monte Carlo "
        r"standard errors for the $t_3$ rows. The improvement probability uses "
        r"the strict comparison $L_{\mathrm{RBPB}}<L_0$; equal-loss cases are not "
        r"counted as improvements.}",
        r"\label{tab:supp-student-t-sensitivity}",
        r"\end{table}",
    ]
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n")


def main():
    groups = load_groups()
    build_figure(groups)
    build_table(groups)
    print(f"wrote {FIGURE_DIR} and {TABLE_PATH}")


if __name__ == "__main__":
    main()
