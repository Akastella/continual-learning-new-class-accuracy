"""Generate paper tables from independently audited R13 integer-count evidence."""
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit/r13_early_retention/analysis_v1/audit.json'
POLICIES = ('sparse_probe', 'dense_probe', 'sparse_plus_actual', 'dense_plus_actual')
LABELS = dict(zip(POLICIES, ('S', 'D', 'S+A', 'D+A')))


def number(x, places=2, signed=False):
    v = Decimal(x['numerator']) / Decimal(x['denominator']) if isinstance(x, dict) else Decimal(str(x))
    v = v.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return format(v, '+f' if signed else 'f')


def main():
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert report['status'] == 'passed'
    a = report['analysis']
    assert len(a['groups']) == 16 and len(a['selections']) == 160
    gen = ROOT / 'paper_r9/generated'
    output = {}

    def table(name, spec, head, rows):
        output[name] = '\\begin{tabular}{' + spec + '}\n\\toprule\n' + head + ' \\\\\n\\midrule\n' + '\n'.join(r + ' \\\\' for r in rows) + '\n\\bottomrule\n\\end{tabular}\n'

    rows = []
    for g in a['groups']:
        rows.append(' & '.join([g['method'].upper(), '20' if g['constraint'] == 'q20' else '20+5', LABELS[g['policy']], str(len(g['available_seeds'])), number(g['metrics']['old']['mean']), number(g['metrics']['current']['mean']), str(g['violations']['constraint_violation'])]))
    table('early_retention_summary.tex', 'lllrrrr', 'History & Rule & Policy & $n$ & Old & Current & Violations', rows)
    for constraint in ('q20', 'q20_rel5'):
        rows = []
        for method in ('sgd', 'er'):
            for seed in range(201, 211):
                fields = [str(seed), method.upper()]
                for policy in POLICIES:
                    r, = [r for r in a['selections'] if (r['method'], r['seed'], r['constraint'], r['policy']) == (method, seed, constraint, policy)]
                    if r['key'] is None:
                        fields.append('NA')
                    else:
                        k = 'A' if r['key'] == 'actual' else r['key'].split('_')[1]
                        flag = '$^{*}$' if r['violations']['constraint_violation'] else ''
                        fields.append(k + ': ' + number(r['test']['old_pct'], 1) + '/' + number(r['test']['current_pct'], 1) + flag)
                rows.append(' & '.join(fields))
        table('early_retention_seeds_' + constraint + '.tex', 'llrrrr', 'Seed & History & S & D & S+A & D+A', rows)
    rows = []
    contrasts = ('dense_probe minus sparse_probe', 'sparse_plus_actual minus sparse_probe', 'dense_plus_actual minus dense_probe')
    for r in a['paired_policy_effects']:
        if r['contrast'] not in contrasts:
            continue
        left, right = r['contrast'].split(' minus ')
        rows.append(' & '.join([r['method'].upper(), '20' if r['constraint'] == 'q20' else '20+5', LABELS[left] + '$-$' + LABELS[right], str(len(r['common_seeds'])), number(r['metrics']['old']['mean'], 3, True), number(r['metrics']['current']['mean'], 3, True), number(r['metrics']['current']['sample_sd'], 3)]))
    table('early_retention_paired.tex', 'lllrrrr', 'History & Rule & Contrast & $n$ & $\\Delta$old & $\\Delta$current & SD current', rows)
    rows = []
    for method in ('sgd', 'er'):
        for seed in range(201, 211):
            anchor = {r['key']: r['scores'] for r in a['anchors'] if (r['method'], r['seed']) == (method, seed)}
            fields = [str(seed), method.upper()]
            for key in ('step_0', 'actual', 'step_628'):
                r = anchor[key]
                fields.append(number(r['old_pct'], 1) + '/' + number(r['current_pct'], 1))
            rows.append(' & '.join(fields))
    table('early_retention_anchors.tex', 'llrrr', 'Seed & History & Pre & Actual & Terminal probe', rows)
    rows = []
    for r in a['conditional_er_minus_sgd']:
        rows.append(' & '.join(['20' if r['constraint'] == 'q20' else '20+5', LABELS[r['policy']], str(len(r['common_seeds'])), number(r['metrics']['old']['mean'], 3, True), number(r['metrics']['current']['mean'], 3, True)]))
    table('early_retention_conditional.tex', 'llrrr', 'Rule & Policy & $n$ & $\\Delta$old & $\\Delta$current', rows)
    for name, content in output.items():
        (gen / name).write_text('% Generated from R13 independent audit; do not edit.\n' + content, encoding='utf-8')
    provenance = {'source': str(SOURCE.relative_to(ROOT)).replace('\\', '/'), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'manifest_sha256': report['manifest_sha256'], 'outputs': {name: hashlib.sha256((gen / name).read_bytes()).hexdigest() for name in output}, 'rounding': 'Exact rational means, Decimal half-up; 2 decimals in summary, 1 in seed display, 3 in paired effects; raw counts retained in source.'}
    (ROOT / 'audit/r13_early_retention/PRESENTATION.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
