# Reproduce the journal figures and numerical tables

The supplied figure inputs are recorded numerical summaries from the experiments. The larger saved predictions and statistical calculations are in the parent code directory. `new_experiments/score_frontier/complete_frontier.csv` includes the 20 validation curves actually plotted; the full per-example score inputs remain in the parent evidence directory.

From this plotting directory, use Python 3.12:

```powershell
python -m pip install -r requirements.txt
python scripts/make_figures.py
python scripts/make_overview.py --export --publication
python scripts/make_supervision_figure.py
python scripts/make_replay_overview.py
python scripts/make_single_pass_table.py
python scripts/make_extra_tables.py
python scripts/make_acquisition_tables.py
python scripts/make_supervision_tables.py
python scripts/make_task7_tables.py
python scripts/make_score_tables.py
python scripts/make_score_figure.py
```

Run the overview command after `make_figures.py`. It reads the ten saved CIFAR-10 seed summaries and produces editable draw.io objects, SVG, PNG, PDF, EPS, and a numerical JSON. Complete export requires draw.io desktop, Comic Sans MS installed locally, Playwright Chromium (`python -m playwright install chromium`), and Poppler (`pdftops`). Supply `--drawio PATH` for a custom installation. Without `--export --publication`, the script rebuilds the draw.io source and numerical JSON.

`make_supervision_figure.py` directly plots the five CIFAR-100 paired histories and validation-selected score results using Matplotlib. Its JSON/CSV inputs are supplied under `new_experiments/cross_group_supervision/`. This figure provides PDF/SVG/EPS, color PNG, and grayscale PNG. The manuscript source package supplies the article and all final vectors.

Open `figures/replay_supervision_overview.drawio` in diagrams.net and export it as a cropped PDF or SVG. The supplied vector files retain the Comic Sans MS labels.
