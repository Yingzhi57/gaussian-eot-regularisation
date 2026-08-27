"""Build the supplementary allocation tables from saved simulation records."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_PATH = REPO_ROOT / "data/allocation/allocation_332_records.csv.gz"
REFERENCE_PATH = REPO_ROOT / "data/allocation/allocation_332_reference.npz"
OUTPUT_PATH = REPO_ROOT / "results/tables/table_332_supplementary.tex"

GEOMETRIES = (
    ("G4-C", "Commuting control"),
    ("G4-N", "Moderate non-commuting"),
    ("G4-K", "Conditioning stress"),
)
RHOS = (0.25, 1.0, 3.0, 7.0, 9.0, 11.0, 15.0)


def ratio_and_mcse(values: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    ratio = float(values.mean() / baseline.mean())
    influence = values - ratio * baseline
    mcse = float(
        influence.std(ddof=1) / np.sqrt(len(baseline)) / baseline.mean()
    )
    return ratio, mcse


def trimmed(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def four_decimals(value: float) -> str:
    if abs(value) < 0.00005:
        value = 0.0
    return f"{value:.4f}"


def table_entry(estimate: float, mcse: float) -> str:
    return rf"${four_decimals(estimate)}\;({four_decimals(mcse)})$"


def load_cells() -> dict[tuple[str, int], dict[str, object]]:
    with gzip.open(RECORDS_PATH, mode="rt", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["failure"] == ""]

    reference = np.load(REFERENCE_PATH, allow_pickle=False)
    cells: dict[tuple[str, int], dict[str, object]] = {}

    for geometry, _ in GEOMETRIES:
        for allocation_index, rho in enumerate(RHOS):
            subset = [
                row
                for row in rows
                if row["geometry"] == geometry
                and int(row["alloc_index"]) == allocation_index
            ]
            expected_n = 10_000 if rho == 0.25 else 5_000
            if len(subset) != expected_n:
                raise ValueError(
                    f"Unexpected number of complete results for {geometry}, "
                    f"rho={rho}: {len(subset)}"
                )

            baseline = np.asarray([float(row["L0"]) for row in subset])
            losses = {
                "kb": np.asarray([float(row["L_KB"]) for row in subset]),
                "dense": np.asarray([float(row["L_dense"]) for row in subset]),
                "rbpb": np.asarray([float(row["L_RBPB"]) for row in subset]),
            }
            ratios = {
                name: ratio_and_mcse(values, baseline)
                for name, values in losses.items()
            }

            gap_values = losses["rbpb"] - losses["dense"]
            gap = float(gap_values.mean() / baseline.mean())
            gap_influence = gap_values - gap * baseline
            gap_mcse = float(
                gap_influence.std(ddof=1)
                / np.sqrt(len(baseline))
                / baseline.mean()
            )

            selected = np.asarray([float(row["m_rbpb"]) for row in subset])
            zero_flags = np.asarray([row["zero"] == "True" for row in subset])
            if not np.array_equal(zero_flags, selected == 0.0):
                raise ValueError(
                    f"Zero-selection indicator mismatch for {geometry}, rho={rho}"
                )

            csv_dense = {float(row["m_dense"]) for row in subset}
            if len(csv_dense) != 1:
                raise ValueError(
                    f"Non-constant dense multiplier for {geometry}, rho={rho}"
                )
            dense_multiplier = float(
                reference[f"{geometry}_{allocation_index}_m_dense"]
            )
            if not np.isclose(dense_multiplier, csv_dense.pop()):
                raise ValueError(
                    f"CSV/NPZ dense multiplier mismatch for {geometry}, rho={rho}"
                )

            cells[(geometry, allocation_index)] = {
                "rho": rho,
                "n": len(subset),
                "m_kb": max(0.0, (11.0 - rho) / (1.0 + rho)),
                "m_dense": dense_multiplier,
                "ratios": ratios,
                "gap": (gap, gap_mcse),
                "selected_q1": float(np.quantile(selected, 0.25)),
                "selected_median": float(np.median(selected)),
                "selected_q3": float(np.quantile(selected, 0.75)),
                "zero_rate": float(zero_flags.mean()),
            }

    if sum(int(cell["n"]) for cell in cells.values()) != 120_000:
        raise ValueError("The 21 cells do not contain 120,000 complete results")
    return cells


def build_risk_table(cells: dict[tuple[str, int], dict[str, object]]) -> list[str]:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"$\rho$ & \shortstack{$m_{\mathrm{KB}}^{(1)}(\rho)$\\gain (MCSE)} & "
        r"\shortstack{$\widehat m_{\mathrm{dense}}$\\gain (MCSE)} & "
        r"\shortstack{RBPB\\gain (MCSE)} & "
        r"\shortstack{RBPB--dense normalised\\gap (MCSE)} \\",
        r"\midrule",
    ]

    for geometry_index, (geometry, description) in enumerate(GEOMETRIES):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{description}}}}} \\")
        for allocation_index, _ in enumerate(RHOS):
            cell = cells[(geometry, allocation_index)]
            ratios = cell["ratios"]
            fields = []
            for rule in ("kb", "dense", "rbpb"):
                ratio, mcse = ratios[rule]
                fields.append(table_entry(100.0 * (1.0 - ratio), 100.0 * mcse))
            gap, gap_mcse = cell["gap"]
            fields.append(table_entry(100.0 * gap, 100.0 * gap_mcse))
            lines.append(
                f"{trimmed(float(cell['rho']))} & " + " & ".join(fields) + r" \\"
            )
        if geometry_index != len(GEOMETRIES) - 1:
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Population-risk results for the $21$ allocation--covariance-"
        r"geometry experiments underlying the left panels of "
        r"Figure~\ref{fig:allocation-covariance-results}. For each decision rule, "
        r"entries are percentage population-risk gains $100(1-\widehat q_p)$ "
        r"relative to $m=0$, with one paired Monte Carlo standard error in "
        r"parentheses; both are in percentage points. The final column is "
        r"$100(\widehat q_{\mathrm{RBPB}}-\widehat q_{\mathrm{dense}})$, with "
        r"its paired Monte Carlo standard error, so a positive value indicates "
        r"greater estimated population risk for RBPB. The number $N_c$ of "
        r"independent evaluation replications with complete paired results is "
        r"$10\,000$ for $\rho=0.25$ and $5000$ otherwise.}",
        r"\label{tab:supp-allocation-risk}",
        r"\end{table}",
    ]
    return lines


def build_selection_table(
    cells: dict[tuple[str, int], dict[str, object]],
) -> list[str]:
    lines = [
        r"\clearpage",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"$\rho$ & $m_{\mathrm{KB}}^{(1)}(\rho)$ & "
        r"$\widehat m_{\mathrm{dense}}$ & "
        r"\shortstack{RBPB $\widehat m_B$\\median [IQR]} & "
        r"\shortstack{RBPB zero-\\selection rate} \\",
        r"\midrule",
    ]

    for geometry_index, (geometry, description) in enumerate(GEOMETRIES):
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{description}}}}} \\")
        for allocation_index, _ in enumerate(RHOS):
            cell = cells[(geometry, allocation_index)]
            selected_summary = (
                f"{trimmed(float(cell['selected_median']))} "
                f"[{trimmed(float(cell['selected_q1']))}, "
                f"{trimmed(float(cell['selected_q3']))}]"
            )
            lines.append(
                f"{trimmed(float(cell['rho']))} & "
                f"{trimmed(float(cell['m_kb']))} & "
                f"{trimmed(float(cell['m_dense']))} & "
                f"{selected_summary} & "
                f"{four_decimals(float(cell['zero_rate']))} " + r"\\"
            )
        if geometry_index != len(GEOMETRIES) - 1:
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Multiplier summaries for the $21$ allocation--covariance-"
        r"geometry experiments underlying the right panels of "
        r"Figure~\ref{fig:allocation-covariance-results}. The first two numeric "
        r"columns give the allocation-specific known-basis leading comparator "
        r"$m_{\mathrm{KB}}^{(1)}(\rho)$ and the independently estimated best "
        r"common multiplier $\widehat m_{\mathrm{dense}}$, respectively. The "
        r"RBPB column reports the median and interquartile range of the selected "
        r"multiplier $\widehat m_B$; the final column is its zero-selection rate.}",
        r"\label{tab:supp-allocation-selection}",
        r"\end{table}",
    ]
    return lines


def main() -> None:
    cells = load_cells()
    output = build_risk_table(cells) + [""] + build_selection_table(cells)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(output) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
