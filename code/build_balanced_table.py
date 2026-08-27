"""Build the balanced Gaussian benchmark table from saved records."""

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/balanced"
OUTPUT_PATH = REPO_ROOT / "results/tables/table_331_gaussian_benchmark.tex"

CASES = [
    ("Case 1", r"$d=5$, $n_1=n_2=100$", "benchmark_331"),
    ("Case 2", r"$d=5$, $n_1=n_2=50$", "benchmark_331_g2a"),
    ("Case 3", r"$d=10$, $n_1=n_2=100$", "benchmark_331_g2b"),
]


def ratio_and_se(Lv, L0v):
    # Paired delta-method MCSE for mean(Lv) / mean(L0v).
    rho = Lv.mean() / L0v.mean()
    infl = Lv - rho * L0v
    return rho, infl.std(ddof=1) / np.sqrt(len(L0v)) / L0v.mean()


lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lccccc@{}}",
    r"\toprule",
    r"Case & \shortstack{$\widehat m_{\mathrm{dense}}$\\"
    r"$\widehat m_{\mathcal M}$} & "
    r"$\widehat q_{5}$ & $\widehat q_{\mathrm{dense}}$ & "
    r"$\widehat q_{\mathrm{RBPB}}$ & "
    r"\shortstack{RBPB $\widehat m_B$\\median [IQR]} \\",
    r"\midrule",
]
preview = []
for name, desc, stem in CASES:
    with (DATA_DIR / f"{stem}_records.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    z = np.load(DATA_DIR / f"{stem}_reference.npz", allow_pickle=False)
    L0 = np.array([float(r["L0"]) for r in rows])
    cols = {}
    for key, col in (("q5", "L_theory"), ("qdense", "L_dense"),
                     ("qrbpb", "L_rbpb")):
        cols[key] = ratio_and_se(
            np.array([float(r[col]) for r in rows]), L0)
    m_hat = np.array([float(r["m_hat"]) for r in rows])
    med = np.median(m_hat)
    q25, q75 = np.quantile(m_hat, .25), np.quantile(m_hat, .75)
    lines.append(rf"{name} & & & & & \\")
    lines.append(
        f"{desc} & {float(z['m_dense']):g}, {float(z['m_grid']):g} & "
        f"{cols['q5'][0]:.4f} ({cols['q5'][1]:.4f}) & "
        f"{cols['qdense'][0]:.4f} ({cols['qdense'][1]:.4f}) & "
        f"{cols['qrbpb'][0]:.4f} ({cols['qrbpb'][1]:.4f}) & "
        f"{med:g} [{q25:g}, {q75:g}] \\\\")
    preview.append(f"{name}: {float(z['m_dense']):g}/{float(z['m_grid']):g}"
                   f"  q5 {cols['q5'][0]:.4f}({cols['q5'][1]:.4f})"
                   f"  qdense {cols['qdense'][0]:.4f}"
                   f"({cols['qdense'][1]:.4f})"
                   f"  qRBPB {cols['qrbpb'][0]:.4f}"
                   f"({cols['qrbpb'][1]:.4f})"
                   f"  m {med:g} [{q25:g}, {q75:g}]")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Balanced Gaussian benchmarks. The rows identify the three cases and",
    r'their designs. For decision rule $p$,',
    r"$\widehat q_p=\overline L^p/\overline L^0$ is its estimated population-risk",
    r"ratio relative to $m=0$; $\widehat q_5$ corresponds to the fixed comparator",
    r"$m=5$, and parentheses give one paired Monte Carlo standard error. The second",
    r"column reports $\widehat m_{\mathrm{dense}}$, the independently estimated best",
    r"common multiplier on the dense mesh, followed by $\widehat m_{\mathcal M}$,",
    r"its fixed-candidate-grid counterpart included only as a grid-discretisation",
    r"diagnostic. The last column reports the median and interquartile range of the",
    r"RBPB selections $\widehat m_B$.}",
    r"\label{tab:rbpb-gaussian-benchmark}",
    r"\end{table}",
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines) + "\n")

print("\n".join(preview))
print(f"\nwritten: {OUTPUT_PATH}")
