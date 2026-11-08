"""Redraw final-task retention curves from the unchanged audited integer counts.

Only presentation changes: unambiguous All tasks legend and visible seed curves.
Requires NumPy, Matplotlib and Pillow; no training or statistical inference.
"""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper_r9/generated/budget_endpoint_counts.csv'
OUT = ROOT / 'figures/r9_budget_retention_v2'
EPOCHS = (4, 8, 16)


def main():
    rows = list(csv.DictReader(SOURCE.open(encoding='utf8', newline='')))
    assert len(rows) == 210
    lookup = {}
    for row in rows:
        key = (row['method'], row['branch'], int(row['seed']), int(row['epoch']))
        assert key not in lookup
        lookup[key] = row
        for group in ('current', 'old', 'all'):
            correct, n = int(row[group+'_correct']), int(row[group+'_n'])
            assert 0 <= correct <= n and n > 0
            assert abs(float(row[group+'_accuracy_pct']) - 100*correct/n) < 1e-10
        assert int(row['current_correct'])+int(row['old_correct']) == int(row['all_correct'])
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.titlesize':8.5, 'axes.labelsize':8, 'xtick.labelsize':7.5,
        'ytick.labelsize':7.5, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False})
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.75), sharex=True, sharey=True)
    for ax, method, title in zip(axs, ('er','derpp'), ('ER actual','DER++ actual')):
        for group, color, marker, style, label in (
            ('current','#0072B2','o','-','Current'),
            ('old','#D55E00','s','--','Old'),
            ('all','#009E73','^',':','All tasks')):
            curves = []
            for seed in range(201,211):
                values = [float(lookup[method,'actual',seed,e][group+'_accuracy_pct']) for e in EPOCHS]
                curves.append(values)
                ax.plot(EPOCHS, values, color=color, ls=style, lw=.5, alpha=.24)
            ax.plot(EPOCHS, np.mean(curves,axis=0), color=color, marker=marker,
                    ls=style, lw=1.6, ms=3, label=label)
        ax.set(title=title, xticks=EPOCHS, ylim=(0,100), yticks=[0,50,100], xlabel='Final-task epoch')
        ax.grid(axis='y', color='.9', lw=.5)
    axs[0].set_ylabel('Accuracy (%)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55,1.04), ncol=3, frameon=False)
    fig.subplots_adjust(left=.10,right=.99,top=.73,bottom=.27,wspace=.18)
    fig.canvas.draw()
    outside=[]
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text():
            box=obj.get_window_extent(fig.canvas.get_renderer())
            if box.width and box.height and (box.x0<0 or box.y0<0 or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1):
                outside.append(obj.get_text())
    assert not outside, outside
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT / ('budget_retention.'+ext), dpi=300,
                    metadata={'Creator':'Matplotlib'} if ext=='pdf' else None)
    plt.close(fig)
    Image.open(OUT/'budget_retention.png').convert('L').save(OUT/'budget_retention_grayscale.png')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'source':SOURCE.relative_to(ROOT).as_posix(), 'source_sha256':sha(SOURCE),
        'generator_sha256':sha(Path(__file__)), 'input_rows':len(rows), 'seeds_per_method':10,
        'epochs':list(EPOCHS), 'plotted_rows':60, 'integer_count_checks':'passed',
        'outside_text':outside, 'scope':'Presentation only; identical existing endpoint values.',
        'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.pdf','.svg')}}
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
