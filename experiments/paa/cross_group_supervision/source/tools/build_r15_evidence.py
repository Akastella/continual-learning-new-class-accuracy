"""Build R15 LaTeX tables from a complete raw-logit analysis; no manual numbers."""
import argparse
import hashlib
import json
from pathlib import Path


def build(source, output):
    payload = source.read_bytes()
    a = json.loads(payload)
    if a['schema'] != 'r15_analysis_v1' or a['n_endpoints'] != 120 or a['seeds'] != list(range(3101, 3106)):
        raise ValueError('Incomplete or unexpected R15 analysis')
    output.mkdir(parents=True, exist_ok=True)
    rows = {(x['method'], x['branch']): x for x in a['endpoint_summary']}
    fmt = lambda x: f"{x['mean']:.2f} $\\pm$ {x['sample_sd']:.2f}"
    table = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Endpoint & Current, seen & Current, ten & Old, seen \\', r'\midrule']
    for method, branch, name in [('sgd', 'actual', 'SGD actual / probe'), ('er', 'actual', 'ER actual'),
                                 ('er', 'probe', 'ER probe'), ('sgd', 'fresh', 'Fresh')]:
        r = rows[method, branch]
        table.append(name + ' & ' + ' & '.join(fmt(r[m]) for m in ['current', 'current_masked', 'old']) + r' \\')
    table += [r'\bottomrule', r'\end{tabular}', '']
    (output/'normal_budget_table.tex').write_text('\n'.join(table), encoding='utf-8')
    table = [r'\begin{tabular}{rrrrc}', r'\toprule',
             r'Seed & Actual & Probe & Masked actual & Joint direction \\', r'\midrule']
    for r in a['paired_seed_contrasts']:
        table.append(f"{r['seed']} & {r['d_actual']:.3f} & {r['d_probe']:.3f} & {r['d_masked']:.3f} & " + ('yes' if r['joint_direction'] else 'no') + r' \\')
    table += [r'\midrule', 'Mean & ' + ' & '.join(f"{a['summary'][k]['mean']:.3f}" for k in ['d_actual', 'd_probe', 'd_masked']) + ' & ' + str(a['joint_direction_seeds']) + r'/5 \\',
              r'\bottomrule', r'\end{tabular}', '']
    (output/'normal_budget_seed_table.tex').write_text('\n'.join(table), encoding='utf-8')
    macros = []
    for suffix, key in [('Actual', 'd_actual'), ('Probe', 'd_probe'), ('Masked', 'd_masked')]:
        macros += [f"\\newcommand{{\\NormalDelta{suffix}}}{{{a['summary'][key]['mean']:.3f}}}",
                   f"\\newcommand{{\\Normal{suffix}Directions}}{{{a['summary'][key]['direction_count']}}}"]
    macros.append(f"\\newcommand{{\\NormalJointDirections}}{{{a['joint_direction_seeds']}}}")
    (output/'normal_budget_macros.tex').write_text('\n'.join(macros)+'\n', encoding='utf-8')
    files = ['normal_budget_table.tex', 'normal_budget_seed_table.tex', 'normal_budget_macros.tex']
    report = {'source_sha256': hashlib.sha256(payload).hexdigest(),
              'source_manifest_sha256': a['manifest_sha256'],
              'files': {n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in files}}
    (output/'normal_budget_evidence.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    build(args.analysis, args.output)
