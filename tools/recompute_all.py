"""Run all CPU statistical checks into one new output folder."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); out=a.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory: '+str(out))
    out.mkdir(parents=True)
    commands=[
        ['tools/recompute_reviewer_subset.py','--output',str(out/'primary_cifar102')],
        ['tools/recompute_ijcnn.py','--output',str(out/'ijcnn')],
        ['tools/recompute_r15_compact.py','--output',str(out/'cifar100')],
        ['analysis/recompute_r17_submission.py','--output',str(out/'weight_and_memory.json')],
        ['analysis/r18_architecture_bridge_compact.py','--root','evidence/r18_bridge','--output',str(out/'resnet32.json')],
        ['tools/recompute_weight_retention.py','--output',str(out/'weight_retention.json')],
        ['tools/recompute_full_history.py','--raw','evidence/history_controls/full_history16/scores','--output',str(out/'full_history16')],
        ['tools/recompute_full_history.py','--raw','evidence/history_controls/seen_classes4/scores','--output',str(out/'seen_classes4')],
        ['tools/compare_validation_calibration.py','--phase','fit','--output',str(out/'validation_calibration')],
        ['tools/compare_validation_calibration.py','--phase','score','--output',str(out/'validation_calibration')],
        ['tools/compare_history_budgets.py','--recomputed-root',str(out),'--output',str(out/'matched_history_budgets.json')],
    ]
    for i,command in enumerate(commands,1):
        print(f'[{i}/{len(commands)}] {command[0]}',flush=True)
        with (out/f'command_{i}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B','-I','-X','utf8',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'PASS: {len(commands)} CPU recomputation commands completed.',flush=True)


if __name__=='__main__':
    main()
