# Entropic regularisation for Gaussian plug-in transport maps

This repository reproduces the simulations in a statistical study of map
estimation from two independent, unpaired samples. The inferential target is
the unregularised Gaussian optimal-transport map. The study asks when entropic
regularisation lowers its repeated-sampling population loss and when it should
be switched off.

The accompanying Rao--Blackwellised parametric bootstrap (RBPB) rule follows
one workflow:

1. estimate the two Gaussian means and covariance matrices;
2. form a data-scaled grid of regularisation levels, including zero;
3. estimate fitted-Gaussian risk by parametric bootstrap;
4. select the smallest risk-minimising multiplier;
5. evaluate the selected map on fresh simulated datasets.

## Repository contents

| Path | Purpose |
|---|---|
| `code/rbpb_selector.py` | RBPB implementation and self-checks |
| `code/benchmark_*.py` | three balanced Gaussian benchmarks |
| `code/allocation_332.py` | 21 allocation--covariance Gaussian designs |
| `code/heavy_333.py` | supplementary Student-t sensitivity study |
| `code/rbpb_*_pilot.py` | candidate-grid and bootstrap-budget pilots |
| `code/build_*.py` | rebuild the report tables and figures from saved records |
| `data/` | simulation-level records and dense-mesh reference curves |
| `results/` | report-facing figures, tables and run summaries |

## Quick reproduction from saved records

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python code/rbpb_selector.py
python code/build_balanced_table.py
python code/build_allocation_figure.py
python code/build_allocation_tables.py
python code/build_student_t_assets.py
```

These commands run the selector checks and rebuild the report-facing numerical
assets without rerunning the Monte Carlo experiments. Outputs are written to
`results/figures/` and `results/tables/`.

## Full simulation workflow

The simulation scripts contain fixed designs, random-number streams and
replication budgets. Run them from `code/` in this order:

```bash
cd code
python rbpb_grid_pilot.py
python rbpb_budget_pilot.py

python benchmark_331.py
python benchmark_g2a_331.py
python benchmark_g2b_331.py

python allocation_332.py
python heavy_333.py
```

The three benchmark scripts are independent and may be run separately. On the
development machine, the allocation study took about 35--55 minutes and the
Student-t study about 10--14 minutes. The scripts write their records and run
summaries to the current directory. The released records in `data/` allow the
tables and figures to be checked immediately. The largest allocation CSV is
stored with lossless gzip compression and is read directly by the portable
`code/build_*.py` scripts.

## Data availability and reproducibility

No observational or personal data are used. Every dataset is simulated from a
distribution specified in the code. The repository includes the Monte Carlo
records, dense-mesh reference curves and the assets used in the report.

Random-number streams for population geometry, reference estimation,
independent evaluation and the inner bootstrap are separated by fixed
`SeedSequence` namespaces. The simulation drivers in `code/` differ from the
files used to create the archived records only in comments and docstrings;
their executable Python syntax is unchanged. The `build_*.py` scripts add
portable repository-relative paths for rebuilding the report assets.

Column definitions are given in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).
File hashes are recorded in `MANIFEST.sha256`.
