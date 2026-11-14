"""Standalone NumPy-only recomputation from the 30 per-example endpoint files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = list(range(3101,3106))
METHODS = ('usual_er','current_mask','mixed_full','mixed_restricted','old_blend05','common_sgd')
METRICS = ('new_all100_pct','new_current10_pct','competition_loss_pp','old_all100_pct','all100_pct')
T4_975 = 2.7764451051977987


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values, exact_mean=None):
    a=np.asarray(values,dtype=np.float64);assert a.shape==(5,) and np.isfinite(a).all()
    mean=float(a.mean()) if exact_mean is None else float(exact_mean)
    sd=float(a.std(ddof=1));se=sd/np.sqrt(5)
    return {'values':a.tolist(),'mean':mean,'sample_sd':sd,'standard_error':float(se),
            'descriptive_t4_95ci':[float(mean-T4_975*se),float(mean+T4_975*se)],
            'positive_seeds':int((a>0).sum()),'zero_seeds':int((a==0).sum()),'negative_seeds':int((a<0).sum())}


def scaled_stats(numerators, denominator):
    # Integer addition precedes division, including exact-zero mean directions.
    return stats([n/denominator for n in numerators],sum(numerators)/(5*denominator))


def compute(root):
    rows=[];metrics={};integer_scores={};identities={};hashes={}
    denominators={'new_all100_pct':10,'new_current10_pct':10,'competition_loss_pp':10,'old_all100_pct':90,'all100_pct':100}
    for seed in SEEDS:
        for name in METHODS:
            path=root/f's{seed}_{name}.npz';hashes[path.name]=digest(path)
            with np.load(path,allow_pickle=False) as d:
                p=d['predictions_all100'];q=d['predictions_current10'];y=d['mapped_labels']
                original=d['original_labels'];order=d['class_order'];indices=d['official_test_indices']
                assert int(d['seed'])==seed and int(d['task'])==9 and str(d['model_name'])==name
                for a in (p,q,y,original,indices): assert a.shape==(10000,) and a.dtype==np.int64
                assert order.shape==(100,) and sorted(order.tolist())==list(range(100))
                assert np.array_equal(indices,np.arange(10000))
                assert np.array_equal(order[y],original)
                assert np.array_equal(np.bincount(y,minlength=100),np.full(100,100))
                assert ((p>=0)&(p<100)).all() and ((q>=90)&(q<100)).all()
                old=y<90;new=~old
                # Restriction preserves the winner whenever all100 predicts a current class.
                assert np.array_equal(q[p>=90],p[p>=90])
                identity=(original.copy(),y.copy(),order.copy(),indices.copy())
                if seed in identities:
                    assert all(np.array_equal(a,b) for a,b in zip(identity,identities[seed]))
                else: identities[seed]=identity
                n_all=int((p==y).sum());n_new=int((p[new]==y[new]).sum())
                n_current=int((q[new]==y[new]).sum());n_old=int((p[old]==y[old]).sum())
                assert n_all==n_new+n_old and n_current>=n_new
                row={'seed':seed,'method':name,'all_correct':n_all,'all_n':10000,
                     'new_all100_correct':n_new,'new_current10_correct':n_current,'new_n':1000,
                     'old_all100_correct':n_old,'old_n':9000,'competition_lost_correct':n_current-n_new}
                rows.append(row)
                metrics[(seed,name)]={'new_all100_pct':n_new/10,'new_current10_pct':n_current/10,
                    'competition_loss_pp':(n_current-n_new)/10,'old_all100_pct':n_old/90,'all100_pct':n_all/100}
                integer_scores[(seed,name)]={'new_all100_pct':n_new,'new_current10_pct':n_current,
                    'competition_loss_pp':n_current-n_new,'old_all100_pct':n_old,'all100_pct':n_all}
                assert abs(metrics[(seed,name)]['all100_pct']-(.9*n_old/90+.1*n_new/10))<1e-12
    methods={name:{m:scaled_stats([integer_scores[(s,name)][m] for s in SEEDS],denominators[m]) for m in METRICS} for name in METHODS}
    pairs={}
    comparisons={'old_restricted_minus_full':('current_mask','usual_er'),
        'mixed_restricted_minus_full':('mixed_restricted','mixed_full'),
        'mixed_full_minus_old_full':('mixed_full','usual_er'),
        'mixed_restricted_minus_old_restricted':('mixed_restricted','current_mask'),
        'blend05_minus_zero_component':('old_blend05','current_mask'),
        'blend05_minus_original_er':('old_blend05','usual_er')}
    for key,(a,b) in comparisons.items():
        pairs[key]={m:scaled_stats([integer_scores[(s,a)][m]-integer_scores[(s,b)][m] for s in SEEDS],denominators[m]) for m in METRICS}
    interaction={m:scaled_stats([integer_scores[(s,'mixed_restricted')][m]-integer_scores[(s,'mixed_full')][m]
                        -integer_scores[(s,'current_mask')][m]+integer_scores[(s,'usual_er')][m] for s in SEEDS],denominators[m]) for m in METRICS}
    accuracy=interaction['new_all100_pct'];gap=interaction['competition_loss_pp']
    h1={'definition':'(mixed restricted - mixed full) - (old-only restricted - old-only full)',
        'new_all100_interaction_positive_mean':accuracy['mean']>0,
        'new_all100_interaction_positive_seeds':accuracy['positive_seeds'],
        'competition_interaction_negative_mean':gap['mean']<0,
        'competition_interaction_negative_seeds':gap['negative_seeds'],
        'supported':bool(accuracy['mean']>0 and accuracy['positive_seeds']>=3 and gap['mean']<0 and gap['negative_seeds']>=3)}
    half_n=[2*integer_scores[(s,'old_blend05')]['new_all100_pct']-integer_scores[(s,'usual_er')]['new_all100_pct'] for s in SEEDS]
    old_n=[integer_scores[(s,'old_blend05')]['old_all100_pct']-integer_scores[(s,'usual_er')]['old_all100_pct'] for s in SEEDS]
    half=[n/20 for n in half_n];old=[n/90 for n in old_n]
    recovered=scaled_stats(half_n,20);old_effect=scaled_stats(old_n,90)
    h2={'coefficient_full_current':.05,'recovery_threshold_fraction_of_baseline':.5,
        'new_minus_half_original_er_pp':recovered,'old_minus_original_er_pp':old_effect,
        'new_recovery_mean_reaches_threshold':recovered['mean']>=0,'new_recovery_seeds_reach_threshold':int((np.asarray(half)>=0).sum()),
        'old_advantage_positive_mean':old_effect['mean']>0,'old_advantage_positive_seeds':old_effect['positive_seeds'],
        'supported':bool(recovered['mean']>=0 and int((np.asarray(half)>=0).sum())>=3 and old_effect['mean']>0 and old_effect['positive_seeds']>=3)}
    summary={'schema':'cross_group_supervision_numpy_v1','independent_units':'five paired seeds',
        'seeds':SEEDS,'n_seeds':5,'old_new_allclass_weights':[.9,.1],
        'methods':methods,'paired_effects':pairs,'restriction_interaction':interaction,'H1':h1,'H2':h2,
        'per_seed_metrics':[{'seed':s,'method':n,**metrics[(s,n)]} for s in SEEDS for n in METHODS],
        'prediction_files_sha256':hashes,
        'scope':'Fixed existing ER histories; current positives in replay change current/old exposure. No claim of equal old exposure across replay policies.',
        'uncertainty':'Mean and sample SD across five paired seeds; t4 intervals are descriptive, not a prespecified significance test.'}
    return rows,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();rows,summary=compute(args.data_dir)
    with (args.data_dir/'recomputed_counts.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.data_dir/'recomputed_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','prediction_files':30,'independent_seeds':5,'H1':summary['H1'],'H2':summary['H2']}))


if __name__=='__main__':main()
