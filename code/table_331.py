# Build the balanced Gaussian benchmark table from saved records and reference
# curves. The table reports risk ratios with paired MCSEs and the distribution
# of RBPB multiplier selections. No simulations are rerun.
# Run: python table_331.py

import csv

import numpy as np

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
    r"Case & $\widehat m_{\mathrm{dense}}/\widehat m_{\mathcal M}$ & "
    r"$\widehat q_{5}$ & $\widehat q_{\mathrm{dense}}$ & "
    r"$\widehat q_{\mathrm{RBPB}}$ & "
    r"\shortstack{RBPB $\widehat m_B$\\median [IQR]} \\",
    r"\midrule",
]
preview = []
for name, desc, stem in CASES:
    rows = [r for r in csv.DictReader(open(f"{stem}_records.csv"))]
    z = np.load(f"{stem}_reference.npz", allow_pickle=True)
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
        f"{desc} & {float(z['m_dense']):g}/{float(z['m_grid']):g} & "
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
    r"\caption{Balanced Gaussian benchmarks. For decision rule $p$, "
    r"$\widehat q_p=\overline L^p/\overline L^0$ is its population-risk "
    r"ratio relative to $m=0$, parentheses give one paired Monte Carlo "
    r"standard error. $\widehat m_{\mathrm{dense}}$ is the independently "
    r"estimated best common multiplier on the dense mesh, and "
    r"$\widehat m_{\mathcal M}$ is the corresponding multiplier restricted "
    r"to the fixed candidate grid, which is included only as a "
    r"grid-discretisation diagnostic. The last column reports the median "
    r"and interquartile range of the RBPB selections $\widehat m_B$.}",
    r"\label{tab:rbpb-gaussian-benchmark}",
    r"\end{table}",
]

open("table_331_gaussian_benchmark.tex", "w").write(
    "\n".join(lines) + "\n"
)

print("\n".join(preview))
print("\nwritten: table_331_gaussian_benchmark.tex")
