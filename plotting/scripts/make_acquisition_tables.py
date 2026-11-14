"""Make the individual-seed classifier table from recorded counts."""
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'new_experiments/head_state/seed_paired.csv').open()))
lines=[r'\begin{tabular}{rrrrr}',r'\toprule',r'Prior epochs & Seed & Inherited & Common initial & Change\\',r'\midrule']
for r in rows:
    values=[r['history_epochs'],r['seed']]+[f"{float(r[k]):+.4f}" for k in ['original_er_minus_sgd_pp','initial_head_er_minus_sgd_pp','difference_in_differences_pp']]
    lines.append(' & '.join(values)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(ROOT/'tables/classifier_seed_values.tex').write_text('\n'.join(lines)+'\n')
print('Wrote classifier-initialization table')
