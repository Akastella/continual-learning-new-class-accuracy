"""Plot new accuracy differences on test-score curves with shared coverage."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("a  Common start", "CIFAR-10", "reservoir_supervision_timing/frontiers.json"),
    ("b  Full streams", "CIFAR-10", "reservoir_supervision_timing/carried_frontiers.json"),
    ("c  Common start", "Fashion-MNIST", "fashion_placement/shared_frontiers.json"),
    ("d  Full streams", "Fashion-MNIST", "fashion_placement/frontiers.json"),
)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.2,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False})
figure,axes=plt.subplots(2,2,figsize=(5.5,4.35),sharex=True)
for axis,(heading,dataset,source) in zip(axes.flat,CASES):
    data=json.loads((ROOT/"new_experiments"/source).read_text(encoding="utf-8"))
    cohort=[row for row in data["rows"] if row["pairs"]>=data["total_pairs"]//4]
    full=[row for row in cohort if row["pairs"]==data["total_pairs"]]
    axis.set_title(heading,loc="left",weight="bold")
    axis.text(.97,.94,dataset,transform=axis.transAxes,ha="right",va="top",size=8.3,color="#394654")
    axis.axhline(0,color="#B0BAC1",linewidth=.8,zorder=0)
    for moment,label,color,marker in (("early","Update 16","#0B6599","-"),
                                       ("late","Task end","#A64E27","--")):
        axis.plot([row["target"] for row in cohort],[row[moment]["mean"] for row in cohort],
                  marker,color=color,linewidth=1.25,alpha=.34)
        axis.plot([row["target"] for row in full],[row[moment]["mean"] for row in full],
                  marker,color=color,linewidth=1.65,label=label)
    axis.set_xlim(0,90)
    axis.set_ylim(-18,28)
    axis.grid(axis="y",color="#D6DCE0",linewidth=.55)
for axis in axes[:,0]:
    axis.set_ylabel("New accuracy change (pp)")
figure.supxlabel("Measured old-class accuracy (%)",y=.095,size=9.3)
figure.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,
              bbox_to_anchor=(.5,.005),handlelength=2.1)
figure.subplots_adjust(left=.145,right=.99,bottom=.19,top=.93,hspace=.34,wspace=.24)
for suffix in ("pdf","svg","png"):
    figure.savefig(ROOT/"figures"/f"timing_frontiers.{suffix}",dpi=200)
