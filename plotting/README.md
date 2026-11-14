# Reproduce the journal figures and numerical tables

The supplied figure inputs are recorded numerical summaries from the experiments. The larger saved predictions and statistical calculations are in the parent code directory. `new_experiments/score_frontier/complete_frontier.csv` includes the 20 validation curves actually plotted; the full per-example score inputs remain in the parent evidence directory.

From this plotting directory, use Python 3.12:

```powershell
python -m pip install -r requirements.txt
python scripts/make_figures.py
python scripts/make_overview.py
python scripts/make_extra_tables.py
python scripts/make_acquisition_tables.py
python scripts/make_supervision_tables.py
python scripts/make_score_tables.py
python scripts/make_score_figure.py
```

Run `make_overview.py` after `make_figures.py` to create the current journal overview. Outputs include editable SVG and PDF figures, PNG previews, grayscale previews, and LaTeX tables. Each point or table cell comes from the supplied inputs. With Poppler installed, EPS files can also be made from each PDF using `pdftops -eps figures/overview.pdf figures/overview.eps`, and similarly for the other figures. The article source package supplies the final editable manuscript and vectors separately.
