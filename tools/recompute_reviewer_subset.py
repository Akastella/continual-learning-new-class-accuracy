"""CPU-only statistical subset: primary integer logs and all R13 scored logits.

This deliberately does NOT call the full checkpoint-lineage auditor. It verifies
only the supplied evidence subset, not training execution or official images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MANIFEST_SHA = 'bb6b8e54d3ffb9ac77a83bdf0fd5f072d5193f5d05763395f45ea2de6a79e86e'
R13_MANIFEST_SHA = '0eb4d79e250749185e8963d11e3c541f7d86eb897d618858569afd36781e7fb8'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summary(values):
    n = len(values)
    m = sum(values, Q()) / n
    v = sum((x - m) ** 2 for x in values) / (n - 1)
    return dict(n=n, mean=float(m), sd=math.sqrt(float(v)),
                mean_fraction=[m.numerator, m.denominator],
                positive=sum(x > 0 for x in values), zero=sum(x == 0 for x in values),
                negative=sum(x < 0 for x in values))


def primary(root):
    manifest = read(root / 'configs/r9_followup/manifest.json')
    archived = read(root / 'results/r9_followup/analysis.json')
    require(sha(root / 'configs/r9_followup/manifest.json') == archived['manifest_sha256'] == PRIMARY_MANIFEST_SHA, 'primary manifest binding')
    expected = {(b, m, s) for b in ('split_cifar10_classil', 'split_cifar10_taskaware')
                for m in ('sgd', 'er') for s in range(201, 211)}
    values = {}
    for rec in manifest['runs']:
        result_path = root / rec['output'] / 'result.json'
        require(sha(result_path) == archived['raw_provenance'][rec['id']]['result_sha256'], 'primary raw hash')
        raw = read(result_path)
        cfg = raw['config']; key = cfg['benchmark'], cfg['method'], cfg['seed']
        require(key in expected and key not in values and cfg['epochs'] == 4 and raw['status'] == 'complete', 'primary run identity/completion')
        require(rec['id'] == f'{key[0]}_{key[1]}_s{key[2]}', 'primary run ID')
        require([t['task'] for t in raw['tasks']] == list(range(5)), 'primary task matrix')
        branches = {}
        for branch in ('fresh', 'probe', 'actual'):
            per_task = []
            for task in raw['tasks'][1:]:
                trace = task[branch]['trace']
                require([t['epoch'] for t in trace] == list(range(5)), 'primary epoch matrix')
                e = trace[4]['evaluation']; c, n = e['correct'], e['n']
                require(type(c) is int and type(n) is int and n == 2000 and 0 <= c <= n, 'primary counts')
                require(abs(e['accuracy_pct'] - 100*c/n) < 1e-10, 'primary stored accuracy')
                per_task.append(Q(100*c, n))
            branches[branch] = sum(per_task, Q()) / 4
        values[key] = branches
    require(set(values) == expected, 'primary missing/duplicate seed')
    rows = []
    for s in range(201, 211):
        sgd, er = (values['split_cifar10_classil', m, s] for m in ('sgd', 'er'))
        require(sgd['fresh'] == er['fresh'] and sgd['probe'] == sgd['actual'], 'shared primary branches')
        rows.append(dict(seed=s, F=sgd['fresh'], a_SGD=sgd['actual'], a_ER=er['actual'],
                         p_SGD=sgd['probe'], p_ER=er['probe'],
                         delta_A=sgd['actual']-er['actual'], delta_H=sgd['probe']-er['probe']))
    # Check the printed exact four-decimal Table 8, rather than a second summary copy.
    tex = (root/'paper_r9/generated/followup_primary_seed_table.tex').read_text()
    require(r'Seed & $F$ & $a_{\rm SGD}$ & $a_{\rm ER}$ & $p_{\rm SGD}$ & $p_{\rm ER}$ & $\Delta A$ & $\Delta H$' in tex, 'Table 8 column semantics')
    printed = [re.findall(r'-?\d+(?:\.\d+)?', line) for line in tex.splitlines() if re.match(r'^2\d\d &', line)]
    require(len(printed) == 10, 'Table 8 missing seeds')
    for row, line in zip(rows, printed):
        require([Q(x) for x in line] == list(row.values()), 'Table 8 numeric mismatch')
    indices = np.random.Generator(np.random.PCG64(2026090901)).integers(10, size=(100000, 10))
    require(hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() == archived['primary']['resample_indices_sha256'], 'bootstrap index identity')
    effects = {}
    for name in ('delta_A', 'delta_H'):
        xs = [r[name] for r in rows]; den = math.lcm(*(x.denominator for x in xs))
        ints = np.array([x.numerator*(den//x.denominator) for x in xs], dtype=np.int64)
        ordered = np.sort(ints[indices].sum(axis=1)); ci = []
        for quantile in (Q(1,80), Q(79,80)):
            position = (len(ordered)-1)*quantile; lo = position.numerator//position.denominator
            w = position-lo
            ci.append(((1-w)*int(ordered[lo])+w*int(ordered[min(lo+1,len(ordered)-1)]))/(den*10))
        effects[name] = dict(summary(xs), interval=[float(v) for v in ci],
                             interval_fractions=[[v.numerator,v.denominator] for v in ci])
        require(np.allclose(effects[name]['interval'], archived['primary']['intervals'][name], atol=1e-12, rtol=0), 'primary interval mismatch')
        require(abs(effects[name]['mean']-archived['primary']['mean'][name]) < 1e-12, 'primary mean mismatch')
    groups = []
    for b in ('split_cifar10_classil', 'split_cifar10_taskaware'):
        for m in ('sgd', 'er'):
            groups.append(dict(benchmark=b, method=m, metrics={br: summary([values[b,m,s][br] for s in range(201,211)]) for br in ('fresh','probe','actual')}))
    table1 = (root/'paper_r9/generated/followup_accuracy_table.tex').read_text()
    require(r'Setting & History & Fresh $F$ & Probe $p$ & Actual $a$' in table1, 'Table 1 column semantics')
    displayed = {}; setting = None
    for line in table1.splitlines():
        cells = [c.strip() for c in line.removesuffix(r'\\').split('&')]
        if len(cells) != 5 or cells[1] not in ('SGD', 'ER'):
            continue
        if cells[0]: setting = cells[0]
        key = setting, cells[1]
        require(key not in displayed, 'Table 1 duplicate labelled row')
        cells = [re.fullmatch(r'\$(-?\d+\.\d+) \\pm (-?\d+\.\d+)\$', c) for c in cells[2:]]
        require(all(c is not None for c in cells), 'Table 1 numeric cell syntax')
        displayed[key] = [c.groups() for c in cells]
    expected_labels = {(b,m) for b in ('Class-incremental','Task-aware') for m in ('SGD','ER')}
    require(set(displayed) == expected_labels, 'Table 1 labelled matrix')
    for g in groups:
        setting = 'Class-incremental' if g['benchmark'].endswith('classil') else 'Task-aware'
        for (mean_text, sd_text), br in zip(displayed[setting,g['method'].upper()], ('fresh','probe','actual')):
            stats = g['metrics'][br]
            require(abs(float(mean_text)-stats['mean']) <= .0050000001 and abs(float(sd_text)-stats['sd']) <= .0050000001, 'Table 1 rounded statistic mismatch')
    return dict(raw_runs=len(values), task_unit='mean of four noninitial tasks within each seed',
                seed_rows=[{k:float(v) if isinstance(v,Q) else v for k,v in row.items()} for row in rows],
                primary=effects, groups=groups,
                bootstrap=dict(generator='PCG64', seed=2026090901, resamples=100000, quantiles=[.0125,.9875], coverage='approximate two-endpoint Bonferroni; conditional on the fixed data'))


def r13(root):
    raw = root/'results/r13_early_retention_v1'
    require(sha(raw/'frozen_manifest.json') == sha(root/'configs/r13_early_retention/manifest.json') == R13_MANIFEST_SHA, 'R13 manifest binding')
    spec = importlib.util.spec_from_file_location('frozen_r13_cpu', root/'analysis/audit_r13_early_retention.py')
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    frozen = read(raw/'frozen_manifest.json')
    require(sha(root/'analysis/audit_r13_early_retention.py') == frozen['source_sha256']['analysis/audit_r13_early_retention.py'], 'frozen independent analysis source')
    inventory = read(raw/'artifacts.json'); sealed = read(raw/'selection_seal.json')
    require(sha(raw/'artifacts.json') == read(raw/'complete.json')['artifacts_sha256'], 'raw inventory binding')
    require(sha(raw/'selection.json') == sealed['selection_sha256'], 'selection seal binding')
    require(set(sealed['validation_files_sha256']) == {rid+'/'+f for rid in audit.RUN_IDS for f in ('training.json','validation.json','validation.npz')}, 'validation seal complete file set')
    for name, expected in sealed['validation_files_sha256'].items():
        require(sha(raw/name) == expected, 'sealed validation/training file: '+name)
    require(sha(raw/'selection_seal.json') == read(raw/'test_started.json')['selection_seal_sha256'], 'test start seal binding')
    require(audit.instant(sealed['created_utc']) < audit.instant(read(raw/'test_started.json')['utc']), 'archived phase order')
    selections = read(raw/'selection.json'); require(set(selections) == set(audit.RUN_IDS), 'R13 history matrix')
    all_runs = []; count = {'validation':0, 'test':0}; originals = {}
    for rid in audit.RUN_IDS:
        seed = int(rid.rsplit('s',1)[1]); method = rid.split('_')[-2]
        cfg_name = 'configs/r9_followup/'+rid+'.json'
        require(sha(root/cfg_name) == frozen['input_sha256'][cfg_name], 'R13 frozen history config')
        cfg = read(root/cfg_name)
        require((cfg['seed'], cfg['method'], cfg['class_order_seed']) == (seed, method, seed+40000), 'R13 history identity')
        order = np.random.Generator(np.random.PCG64(seed+40000)).permutation(10)
        rd = raw/rid; phases = {}
        choices = None
        for phase in ('validation', 'test'):
            for suffix in ('json', 'npz'):
                name = rid+'/'+phase+'.'+suffix
                require(sha(raw/name) == inventory[name], 'R13 archived raw binding: '+name)
                if phase == 'validation':
                    require(sha(raw/name) == sealed['validation_files_sha256'][name], 'validation seal member')
            meta = read(rd/(phase+'.json'))
            require(meta['run_id'] == rid and meta['npz_sha256'] == sha(rd/(phase+'.npz')), 'raw metadata identity')
            if phase == 'test':
                require(meta['selection_seal_sha256'] == sha(raw/'selection_seal.json'), 'test endpoint seal')
            wanted = {f'step_{s}' for s in audit.DENSE}|{'actual'} if phase == 'validation' else audit.ANCHORS|{v for c in choices.values() for v in c.values() if v is not None}
            records = audit.records_by_key(meta['records'], wanted)
            with np.load(rd/(phase+'.npz'), allow_pickle=False) as data:
                labels, indices = data['labels'], data['indices']
                require(labels.shape == indices.shape == ((1000,) if phase == 'validation' else (2000,)), 'phase sample size')
                require(len(np.unique(indices)) == len(indices), 'duplicate image indices')
                original = order[labels]
                if phase in originals:
                    require(np.array_equal(original, originals[phase][0]) and np.array_equal(indices, originals[phase][1]), 'cross-history label mapping or indices')
                else: originals[phase] = original.copy(), indices.copy()
                if phase == 'test': require(np.array_equal(indices,np.arange(2000)), 'official test index sequence')
                # Original image arrays are not included: this checks archived labels,
                # not an independent download of official ground-truth labels.
                rebuilt = audit.check_npz(rd/(phase+'.npz'), records, indices, labels)
            count[phase] += len(rebuilt); phases[phase] = rebuilt
            if phase == 'validation':
                choices = {c:{p:audit.select(rebuilt,p,c) for p in audit.POLICIES} for c in audit.CONSTRAINTS}
                require(choices == selections[rid], 'validation selection mismatch')
        all_runs.append(dict(run_id=rid, seed=seed, method=method, validation=phases['validation'], test=phases['test'], selection=choices))
    recomputed = audit.summarize(all_runs)
    require(recomputed == read(root/'audit/r13_early_retention/analysis_v1/audit.json')['analysis'], 'full R13 numerical analysis mismatch')
    require(count == {'validation':500,'test':89}, 'R13 incomplete endpoint set')
    return dict(endpoints=count, analysis=recomputed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root.resolve(); out = args.output.resolve()
    require(not out.exists(), 'Output must be new')
    main_results = primary(root); policy_results = r13(root)
    out.mkdir(parents=True)
    result = dict(status='passed_statistical_subset', scope='Primary integer-count statistics and R13 archived-logit statistics only; not full evidence audit, official-image verification, forward inference or training replay.',
                  numpy_version=np.__version__, primary=main_results,
                  r13=dict(endpoints=policy_results['endpoints'], selections=160, numerical_analysis_exact=True))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'r13_analysis.json').write_text(json.dumps(policy_results['analysis'],indent=2)+'\n',encoding='utf-8')
    with (out/'primary_seed_values.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(main_results['seed_rows'][0]));w.writeheader();w.writerows(main_results['seed_rows'])
    print(json.dumps({k:result[k] for k in ('status','scope','r13')},indent=2))


if __name__ == '__main__':
    main()
