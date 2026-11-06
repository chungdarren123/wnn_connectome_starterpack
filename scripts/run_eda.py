"""Reproducible, sequence-aware EDA; never reads confirmation/reserved rows.

Uses full file metadata, all rows in 256 seeded training sequences, and all
64 development sequences. Row-level multivariate diagnostics use 512 sampled
post-warm-up rows per sequence. No model training for the competition occurs.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
FEATURES = [f'i{i}_{g}{j}' for i in range(2) for g,n in [('p',22),('v',22),('dp',4),('dv',4)] for j in range(n)] + [f'a{i}' for i in range(8)]
TARGETS = ['t0', 't1']
SEED = 20261101
LAGS = [1, 4, 16, 64, 256, 1024]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_table(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def run_lengths(flags):
    change = np.flatnonzero(np.diff(np.r_[False, flags, False].astype(np.int8)))
    return (change[1::2] - change[::2]).tolist()


def corr(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summary_y(y):
    result = {}
    for k, name in enumerate(TARGETS):
        z = y[:, k]
        finite = z[np.isfinite(z)]
        if not len(finite):
            result[name] = {'rows': len(z), 'finite_rows': 0}
            continue
        clipped = np.clip(finite, -2, 2)
        w = np.abs(clipped)
        order = np.sort(w)
        total = float(w.sum(dtype=np.float64))
        result[name] = {
            'rows': len(z), 'finite_rows': len(finite), 'mean': float(finite.mean(dtype=np.float64)),
            'std': float(finite.std(dtype=np.float64)), 'min': float(finite.min()), 'max': float(finite.max()),
            'quantiles': dict(zip(['p01','p05','p25','p50','p75','p95','p99'], np.quantile(finite,[.01,.05,.25,.5,.75,.95,.99]).tolist())),
            'zero_fraction': float(np.mean(finite == 0)), 'negative_fraction': float(np.mean(finite < 0)),
            'outside_clip_fraction': float(np.mean(np.abs(finite) > 2)),
            'clipped_abs_weight_sum': total,
            'largest_10pct_rows_weight_share': float(order[-max(1,len(order)//10):].sum(dtype=np.float64)/total) if total else 0,
            'weight_effective_sample_size': float(total**2/np.square(w.astype(np.float64)).sum()) if total else 0,
        }
    result['target_correlation'] = corr(y[:,0], y[:,1])
    return result


def inspect_metadata(path):
    pf = pq.ParquetFile(path)
    counts = Counter(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
    codecs = Counter(pf.metadata.row_group(i).column(0).compression for i in range(pf.num_row_groups))
    return {
        'path': path.name, 'bytes': path.stat().st_size, 'rows': pf.metadata.num_rows,
        'row_groups': pf.num_row_groups, 'columns': len(pf.schema_arrow),
        'row_group_sizes': dict(counts), 'compression_first_column': dict(codecs),
        'schema': [{'name':f.name,'type':str(f.type),'nullable':f.nullable} for f in pf.schema_arrow],
    }


def analyze_sequences(pf, indices, name, output):
    rng = np.random.default_rng(SEED + (1 if name == 'train' else 2))
    count = np.zeros(112, np.int64)
    missing = np.zeros(112, np.int64)
    sums = np.zeros(112)
    squares = np.zeros(112)
    low = np.full(112, np.inf)
    high = np.full(112, -np.inf)
    changed = np.zeros(112, np.int64)
    changed_den = np.zeros(112, np.int64)
    arrow_nulls = np.zeros(112, np.int64)
    rows = []
    sampled = []
    all_y = []
    scored_y = []
    scored_flags = []
    mask_runs = []
    examples = None
    acf = {lag: [] for lag in LAGS}
    lag_corr_acc = {lag: [np.zeros(112),np.zeros(112),np.zeros(2),np.zeros(2),np.zeros((112,2)),0] for lag in [0,1,4,16,64,256]}
    sequence_ids = set()
    for position, group in enumerate(indices):
        table = pf.read_row_group(group, use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError('Incomplete sequence')
        seq_ids = table['seq_ix'].to_numpy()
        seq = int(seq_ids[0])
        if not np.all(seq_ids == seq) or seq in sequence_ids:
            raise ValueError('Sequence identity error')
        sequence_ids.add(seq)
        step = table['step_in_seq'].to_numpy()
        need = table['need_prediction'].to_numpy().astype(bool)
        if not np.array_equal(step,np.arange(20_000)) or not np.array_equal(need,step >= 99):
            raise ValueError('Step/warm-up contract violation')
        x = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in TARGETS]).astype(np.float32)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite targets')
        arrow_nulls += np.array([table[c].null_count for c in FEATURES])
        ok = np.isfinite(x)
        count += ok.sum(axis=0)
        missing += (~ok).sum(axis=0)
        clean = np.where(ok,x,0).astype(np.float64)
        sums += clean.sum(axis=0)
        squares += np.square(clean).sum(axis=0)
        low = np.minimum(low,np.where(ok,x,np.inf).min(axis=0))
        high = np.maximum(high,np.where(ok,x,-np.inf).max(axis=0))
        adjacent_ok = ok[1:] & ok[:-1]
        changed += ((x[1:] != x[:-1]) & adjacent_ok).sum(axis=0)
        changed_den += adjacent_ok.sum(axis=0)
        eligible_y = y[need]
        all_y.append(eligible_y)
        scored = table['is_scored'].to_numpy().astype(bool) if name == 'development' else None
        if scored is not None:
            if scored[:99].any():
                raise ValueError('Scored warm-up rows')
            scored_y.append(y[need & scored])
            scored_flags.append(scored[need])
            mask_runs.extend(run_lengths(scored & need))
        pick = np.sort(rng.choice(np.arange(1024,20_000),512,replace=False))
        sample_mask = scored[pick] if scored is not None else np.ones(len(pick),bool)
        sampled.append((x[pick],y[pick],np.full(len(pick),seq,np.int64),pick,sample_mask))
        for lag in LAGS:
            acf[lag].append([corr(y[lag:,k], y[:-lag,k]) for k in range(2)])
        for lag, acc in lag_corr_acc.items():
            a = x[pick-lag].astype(np.float64)
            b = y[pick].astype(np.float64)
            valid = np.isfinite(a).all(axis=1)
            a,b = a[valid],b[valid]
            acc[0] += a.sum(axis=0)
            acc[1] += np.square(a).sum(axis=0)
            acc[2] += b.sum(axis=0)
            acc[3] += np.square(b).sum(axis=0)
            acc[4] += a.T @ b
            acc[5] += len(a)
        info = {'dataset':name,'source_row_group':int(group),'seq_ix':seq,
                'rows':len(y),'scored_rows':int((scored & need).sum()) if scored is not None else None,
                'feature_nonfinite':int((~ok).sum()),
                'feature_constant_columns':int(np.sum(np.std(clean,axis=0) < 1e-12))}
        for k,c in enumerate(TARGETS):
            info[c+'_mean'] = float(eligible_y[:,k].mean(dtype=np.float64))
            info[c+'_std'] = float(eligible_y[:,k].std(dtype=np.float64))
            info[c+'_zero_fraction'] = float(np.mean(eligible_y[:,k] == 0))
            info[c+'_unique_values'] = len(np.unique(eligible_y[:,k]))
            info[c+'_outside_clip_fraction'] = float(np.mean(np.abs(eligible_y[:,k]) > 2))
            if scored is not None:
                info[c+'_scored_weight'] = float(np.abs(np.clip(y[need & scored,k],-2,2)).sum(dtype=np.float64))
        rows.append(info)
        if examples is None:
            examples = {'seq_ix':seq,'step':step.tolist(),'t0':y[:,0].tolist(),'t1':y[:,1].tolist(),
                        'features':{c:x[:,FEATURES.index(c)].tolist() for c in ['i0_p0','i0_v0','i0_dp0','i1_p0','a0','a3']},
                        'is_scored':scored.tolist() if scored is not None else None}
        if (position+1)%16 == 0 or position+1 == len(indices):
            print(f'{name}: analyzed {position+1}/{len(indices)} sequences',flush=True)
    xs,ys,groups,steps,mask = [np.concatenate([s[i] for s in sampled]) for i in range(5)]
    mean = sums/np.maximum(count,1)
    std = np.sqrt(np.maximum(0,squares/np.maximum(count,1)-mean**2))
    quant = np.nanquantile(np.where(np.isfinite(xs),xs,np.nan),[0,.01,.05,.25,.5,.75,.95,.99,1],axis=0)
    feature_rows = []
    for j,c in enumerate(FEATURES):
        vals,freq = np.unique(xs[np.isfinite(xs[:,j]),j],return_counts=True)
        mode_index = int(np.argmax(freq)) if len(freq) else None
        entry = {'dataset':name,'feature':c,'finite_count':int(count[j]),'nonfinite_count':int(missing[j]),
                 'arrow_null_count':int(arrow_nulls[j]),'mean':float(mean[j]),'std':float(std[j]),
                 'min':finite_or_none(low[j]),'max':finite_or_none(high[j]),
                 'change_fraction':float(changed[j]/max(1,changed_den[j])),
                 'sample_unique_values':len(vals),'sample_mode':float(vals[mode_index]) if mode_index is not None else None,
                 'sample_mode_fraction':float(freq[mode_index]/len(xs)) if mode_index is not None else 0,
                 'sample_fraction_below_minus4':float(np.mean(xs[:,j] < -4))}
        entry.update({f'sample_{q}':finite_or_none(quant[i,j]) for i,q in enumerate(['min','p01','p05','p25','p50','p75','p95','p99','max'])})
        feature_rows.append(entry)
    lag_rows = []
    for lag,(sx,sxx,sy,syy,sxy,n) in lag_corr_acc.items():
        numerator = sxy-np.outer(sx,sy)/n
        den = np.sqrt(np.maximum(0,sxx-sx*sx/n)[:,None]*np.maximum(0,syy-sy*sy/n)[None,:])
        r = np.divide(numerator,den,out=np.zeros_like(numerator),where=den>1e-12)
        for j,c in enumerate(FEATURES):
            lag_rows.append({'dataset':name,'lag':lag,'feature':c,'t0_correlation':float(r[j,0]),'t1_correlation':float(r[j,1]),'rows':n})
    summary = {
        'sequences':len(indices),'rows':len(indices)*20_000,'multivariate_sample_rows':len(xs),
        'sample_step_range':[1024,19999], 'source_row_groups':indices,'sequence_ids':sorted(sequence_ids),
        'feature_nonfinite_count':int(missing.sum()),'feature_arrow_null_count':int(arrow_nulls.sum()),
        'target_statistics_post_warmup':summary_y(np.concatenate(all_y)),
        'target_autocorrelation_median_by_sequence':{str(lag):[finite_or_none(np.nanmedian([a[k] if a[k] is not None else np.nan for a in acf[lag]])) for k in range(2)] for lag in LAGS},
    }
    if scored_y:
        concatenated = np.concatenate(scored_y)
        summary['target_statistics_scored'] = summary_y(concatenated)
        summary['scored_rows'] = len(concatenated)
        summary['scored_fraction_post_warmup'] = float(np.mean(np.concatenate(scored_flags)))
        summary['scored_run_lengths'] = {'runs':len(mask_runs),'median':float(np.median(mask_runs)),
                                         'p90':float(np.quantile(mask_runs,.9)),'max':int(max(mask_runs))}
    csv_table(output/f'{name}_features.csv',feature_rows)
    csv_table(output/f'{name}_sequences.csv',rows)
    csv_table(output/f'{name}_lag_correlations.csv',lag_rows)
    return {'summary':summary,'feature_rows':feature_rows,'sequence_rows':rows,
            'x':xs,'y':ys,'groups':groups,'steps':steps,'mask':mask,'example':examples,'lag_rows':lag_rows,
            'eligible_y':np.concatenate(all_y),'scored_y':np.concatenate(scored_y) if scored_y else None}


def classifier_diagnostic(x,y,groups,seed):
    # Feature-only models; splitting by sequence prevents adjacent-row leakage.
    splitter = GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed)
    train,test = next(splitter.split(x,y,groups))
    finite_x = np.where(np.isfinite(x),x,np.nan)
    keep = np.isfinite(finite_x).all(axis=1)
    train,test = train[keep[train]],test[keep[test]]
    result = {'training_sequences':len(np.unique(groups[train])),'heldout_sequences':len(np.unique(groups[test])),
              'training_rows':len(train),'heldout_rows':len(test),
              'heldout_positive_fraction':float(y[test].mean())}
    if min(len(np.unique(y[train])),len(np.unique(y[test]))) < 2:
        result['skipped'] = 'One class in sequence-held-out split'
        return result
    models = {
        'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',random_state=seed)),
        'shallow_boosting':HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1,random_state=seed),
    }
    for name,model in models.items():
        model.fit(finite_x[train],y[train])
        score = model.predict_proba(finite_x[test])[:,1]
        result[name+'_heldout_auc'] = float(roc_auc_score(y[test],score))
    return result


def plot_all(train,dev,shift,pca,output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    images = []
    def save(name,fig):
        path = output/(name+'.png')
        fig.savefig(path,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        images.append(path)
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for k,c in enumerate(TARGETS):
        for data,label,color in [(train['eligible_y'],'Train sample','#3366aa'),(dev['eligible_y'],'Development all eligible','#e69528'),(dev['scored_y'],'Development scored','#16856c')]:
            axes[0,k].hist(data[:,k],bins=np.linspace(-4,4,81),density=True,histtype='step',label=label,color=color)
        axes[0,k].set(title=f'{c}: target distribution',xlabel='Target (display limited to [-4, 4])',ylabel='Density')
        axes[0,k].axvline(-2,color='grey',ls='--'); axes[0,k].axvline(2,color='grey',ls='--')
        axes[0,k].legend(fontsize=8)
        for data,label,color in [(train['eligible_y'],'Train sample','#3366aa'),(dev['scored_y'],'Development scored','#16856c')]:
            w = np.sort(np.abs(np.clip(data[:,k],-2,2)))[::-1].astype(np.float64)
            index = np.linspace(0,len(w)-1,min(1000,len(w)),dtype=int)
            axes[1,k].plot((index+1)/len(w),np.cumsum(w)[index]/w.sum(),label=label,color=color)
        axes[1,k].set(title=f'{c}: concentration of metric weight',xlabel='Fraction of rows, highest weights first',ylabel='Fraction of total target weight')
        axes[1,k].plot([0,1],[0,1],ls=':',color='grey'); axes[1,k].legend(fontsize=8)
    fig.tight_layout(); save('01_targets_and_metric_weights',fig)
    top = shift[:20]
    fig,axes = plt.subplots(1,2,figsize=(12,7))
    axes[0].barh([r['feature'] for r in top][::-1],[r['ks_distance'] for r in top][::-1],color='#3366aa')
    axes[0].set(title='Largest train/development distribution differences',xlabel='KS distance (effect size; no IID p-values)')
    c = top[0]['feature']; j=FEATURES.index(c)
    vals=np.r_[train['x'][:,j],dev['x'][:,j]]
    vals=vals[np.isfinite(vals)]
    lim=np.quantile(vals,[.005,.995])
    for data,label,color in [(train,'Train sample','#3366aa'),(dev,'Development','#e69528')]:
        axes[1].hist(data['x'][:,j],bins=np.linspace(lim[0],lim[1],61),density=True,histtype='step',label=label,color=color)
    axes[1].set(title=f'{c}: strongest marginal shift',xlabel='Feature value (central 99% display)',ylabel='Density'); axes[1].legend()
    fig.tight_layout(); save('02_distribution_shift',fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    example=dev['example']; end=2000
    for c in ['i0_p0','i1_p0','a0']:
        axes[0,0].plot(example['step'][:end],example['features'][c][:end],label=c,lw=.7)
    axes[0,0].set(title=f'Development sequence {example["seq_ix"]}: features',xlabel='Sequence step',ylabel='Anonymized feature value');axes[0,0].legend(fontsize=8)
    for c in TARGETS:
        axes[0,1].plot(example['step'][:end],example[c][:end],label=c,lw=.8)
    mask=np.array(example['is_scored'][:end]); axes[0,1].scatter(np.arange(end)[mask],np.zeros(mask.sum()),s=5,color='black',label='Scored rows',alpha=.4)
    axes[0,1].set(title='Targets and scoring mask (first 2,000 steps)',xlabel='Sequence step',ylabel='Target');axes[0,1].legend(fontsize=8)
    for data,label,style in [(train,'Train','-'),(dev,'Development','--')]:
        for k,c in enumerate(TARGETS):
            acf=data['summary']['target_autocorrelation_median_by_sequence']
            axes[1,0].plot(LAGS,[acf[str(lag)][k] for lag in LAGS],style,marker='o',label=f'{label} {c}')
    axes[1,0].set(xscale='log',title='Median within-sequence target autocorrelation',xlabel='Lag in rows',ylabel='Pearson correlation');axes[1,0].legend(fontsize=8)
    rows=dev['sequence_rows']
    axes[1,1].hist([r['scored_rows']/19901 for r in rows],bins=16,color='#16856c')
    axes[1,1].set(title='Scored-row fraction varies by development sequence',xlabel='Scored / required rows',ylabel='Sequences')
    fig.tight_layout();save('03_temporal_structure_and_mask',fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for k,c in enumerate(TARGETS):
        xx=np.where(np.isfinite(train['x']),train['x'],0)
        r=np.corrcoef(np.column_stack([xx,train['y'][:,k]]),rowvar=False)[-1,:112]
        ids=np.argsort(np.abs(r))[-15:]
        axes[0,k].barh([FEATURES[i] for i in ids],r[ids],color='#3366aa')
        axes[0,k].set(title=f'Train: strongest marginal correlations with {c}',xlabel='Unweighted Pearson; sampled eligible rows')
        for data,label,color in [(train,'Train','#3366aa'),(dev,'Development','#e69528')]:
            lags=sorted(set(row['lag'] for row in data['lag_rows']))
            best=[max(abs(row[c+'_correlation']) for row in data['lag_rows'] if row['lag']==lag) for lag in lags]
            axes[1,k].plot(lags,best,marker='o',label=label,color=color)
        axes[1,k].set(xscale='symlog',title=f'Strongest marginal lagged-feature relationship: {c}',xlabel='Past input lag (0 = current row)',ylabel='Maximum absolute feature correlation')
        axes[1,k].legend()
    fig.tight_layout(); save('04_feature_target_relationships',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    xx=np.where(np.isfinite(train['x']),train['x'],0)
    correlation=np.corrcoef(xx,rowvar=False)
    axes[0].imshow(correlation,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    axes[0].set(title='Train feature correlation matrix',xlabel='Official feature index',ylabel='Official feature index')
    for value in [51.5,103.5]:
        axes[0].axhline(value,color='black',lw=.6);axes[0].axvline(value,color='black',lw=.6)
    axes[1].plot(np.arange(1,len(pca)+1),np.cumsum(pca),color='#3366aa')
    axes[1].set(title='Train standardized PCA',xlabel='Principal components',ylabel='Cumulative explained variance',ylim=(0,1.01))
    fig.tight_layout();save('05_redundancy_and_pca',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for data,label,color in [(train,'Train sample','#3366aa'),(dev,'Development','#e69528')]:
        rows=data['feature_rows']
        axes[0].plot([r['change_fraction'] for r in rows],label=label,color=color)
        axes[1].plot([r['sample_fraction_below_minus4'] for r in rows],label=label,color=color)
    axes[0].set(title='How often each feature changes',xlabel='Official feature index',ylabel='Fraction of adjacent finite rows that differ')
    axes[1].set(title='Potential sentinel/extreme negative values',xlabel='Official feature index',ylabel='Sample fraction below -4 (not confirmed missingness)')
    for ax in axes:
        ax.legend()
        for value in [51.5,103.5]:ax.axvline(value,color='grey',ls=':')
    fig.tight_layout();save('06_feature_activity_and_extremes',fig)
    return images


def build_html(output,summary,train,dev,shift,images):
    def table(records,keys):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in records)+'</tbody></table>'
    def num(x):return f'{x:.4f}'
    t=train['summary']; d=dev['summary']
    domain=summary['domain_classification']; mask=summary['scoring_mask_classification']
    observations=[
        f'Analyzed all rows in {t["sequences"]} training sequences ({t["rows"]:,} rows) and all {d["sequences"]} fixed development sequences ({d["rows"]:,} rows). Confirmation and the other validation rows were not read for distribution analysis.',
        f'Feature nonfinite counts in analyzed rows: train {t["feature_nonfinite_count"]:,}; development {d["feature_nonfinite_count"]:,}. Arrow null counts: train {t["feature_arrow_null_count"]:,}; development {d["feature_arrow_null_count"]:,}. This is not a full-file integrity claim.',
        f'Development selects {d["scored_rows"]:,} rows, {100*d["scored_fraction_post_warmup"]:.2f}% of required prediction rows. Scored runs have median length {d["scored_run_lengths"]["median"]} rows. Scoring-mask information is never an inference input.',
        f'Largest marginal feature shift: {shift[0]["feature"]}, KS distance {shift[0]["ks_distance"]:.4f}. Sequence-held-out domain classifier AUC: logistic {domain.get("logistic_heldout_auc",float("nan")):.4f}, shallow boosting {domain.get("shallow_boosting_heldout_auc",float("nan")):.4f}. This indicates separability of these samples, not knowledge of the test distribution.',
        f'Sequence-held-out development mask classifier AUC: logistic {mask.get("logistic_heldout_auc",float("nan")):.4f}, shallow boosting {mask.get("shallow_boosting_heldout_auc",float("nan")):.4f}. A single split is exploratory and is not evidence that a hidden test mask follows the same mechanism.',
    ]
    for c in TARGETS:
        target=d['target_statistics_scored'][c]
        observations.append(f'{c} on scored development rows: zero fraction {100*target["zero_fraction"]:.2f}%; targets beyond +/-2 {100*target["outside_clip_fraction"]:.2f}%; largest 10% of rows account for {100*target["largest_10pct_rows_weight_share"]:.2f}% of metric weight. Weight-based effective sample size {target["weight_effective_sample_size"]:,.0f} ignores temporal dependence and is not an uncertainty estimate.')
    target_rows=[]
    for data,label,field in [(t,'train eligible','target_statistics_post_warmup'),(d,'development eligible','target_statistics_post_warmup'),(d,'development scored','target_statistics_scored')]:
        for c in TARGETS:
            stats=data[field][c]
            target_rows.append({'population':label,'target':c,'rows':stats['rows'],'mean':num(stats['mean']),'std':num(stats['std']),'zero_fraction':num(stats['zero_fraction']),'outside_clip_fraction':num(stats['outside_clip_fraction'])})
    body='''<!doctype html><html><head><meta charset="utf-8"><title>Connectome EDA v1</title><style>
body{max-width:1150px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#172b3a;background:#fafbfd}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:8px;border:1px solid #d5dce3;text-align:left}th{background:#e8eef5}img{max-width:100%;background:white;border:1px solid #d5dce3;margin:18px 0}code{background:#edf1f4;padding:2px 5px}.scope{background:#e8f1fa;padding:18px;border-left:4px solid #3366aa}li{margin-bottom:9px}</style></head><body>'''
    body+='<h1>Connectome exploratory data analysis — v1</h1>'
    body+='<p class="scope"><b>Scope:</b> exact metadata for source files; full-row summaries of 256 randomly selected training sequences and 64 fixed development sequences; 512 post-warm-up rows per sequence for multivariate diagnostics. Confirmation and reserved validation rows remain untouched. This report contains descriptive analysis, not validated model improvements.</p>'
    body+='<h2>Measured findings</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in observations)+'</ul>'
    body+='<h2>Dataset metadata</h2>'+table([{'file':m['path'],'rows':m['rows'],'sequences':m['row_groups'],'columns':m['columns'],'GB':round(m['bytes']/1e9,3)} for m in summary['metadata']],['file','rows','sequences','columns','GB'])
    body+='<h2>Target distributions</h2>'+table(target_rows,list(target_rows[0]))
    body+='<h2>Figures</h2>'
    for image in images:
        body+='<img alt="'+html.escape(image.stem)+'" src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'">'
    body+='<h2>Largest feature distribution differences</h2>'+table([{k:round(v,5) if isinstance(v,float) else v for k,v in r.items()} for r in shift[:25]],list(shift[0]))
    body+='<h2>Interpretation and next experiments</h2><ul>'
    for text in [
        'Prioritize a causal change/EMA feature baseline and a trainable GRU. Target autocorrelation and lagged-feature correlations describe dependence; neither is proof of an exploitable causal signal.',
        'Audit extreme repeated values before adding sentinel indicators. Values below -4 are only a diagnostic threshold; the undocumented feature transforms do not establish that these values mean missing data.',
        'Compare all-eligible and scored target distributions. Their difference describes selection, not an instruction to predict only on visible scored rows.',
        'Use train/development shift diagnostics to motivate robustness checks. Do not assume the hidden test follows development or train a mask-specific predictor without independent evidence.',
        'Investigate target discretization using per-sequence unique counts before proposing a classification head. No true target horizon, tick size, raw price, or raw spread is inferred by this report.',
        'Fit all model preprocessing on training data. Sequence IDs only group rows; they are not chronological features. Never shuffle rows within a model sequence or use future rows in causal transforms.',
        'Correlations, KS distances and PCA use sampled rows. Nearby rows remain dependent. No IID p-values or confidence claims are supplied. Group-held-out classifiers use one split and need repeated group splits before firm conclusions.',
        'The fixed development set is now used for exploration. Keep confirmation out of training and tuning; full local validation is not independent of development.',
    ]: body+='<li>'+html.escape(text)+'</li>'
    body+='</ul><h2>Reproduction and outputs</h2><p><code>python scripts/run_eda.py</code> from the starter-pack root. Outputs are immutable: choose a new <code>--output</code> directory for another run. See <code>summary.json</code>, <code>feature_shift.csv</code>, per-dataset feature/sequence/lag tables, the correlation matrix, and <code>environment.txt</code>. No source data, trained challenge models, or raw sampled arrays are saved in this report folder.</p>'
    body+='<p>EDA seed: '+str(SEED)+'. Source selection and hashes are in summary.json. No code or data is uploaded.</p></body></html>'
    (output/'report.html').write_text(body,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('experiments/eda/eda_v1'))
    args=parser.parse_args();root=args.root.resolve()
    output=args.output if args.output.is_absolute() else root/args.output
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    started=time.perf_counter()
    manifest_path=root/'experiments/splits/validation_v1.json'
    manifest=json.loads(manifest_path.read_text())
    pf=pq.ParquetFile(root/'datasets/train.parquet')
    train_indices=sorted(random.Random(SEED).sample(range(pf.num_row_groups),256))
    print('Reading metadata; only training sample and development rows will be analyzed.',flush=True)
    metadata=[inspect_metadata(root/'datasets'/name) for name in ['train.parquet','valid.parquet','valid_mask.parquet']]
    devpath=root/manifest['subsets']['development']['path']
    if hashlib.sha256(devpath.read_bytes()).hexdigest()!=manifest['subsets']['development']['sha256']:
        raise ValueError('Development subset checksum mismatch')
    train=analyze_sequences(pf,train_indices,'train',output)
    dev=analyze_sequences(pq.ParquetFile(devpath),list(range(64)),'development',output)
    if set(train['summary']['sequence_ids']) & set(dev['summary']['sequence_ids']):
        raise ValueError('Training/development sequence IDs overlap')
    expected=set(r['seq_ix'] for r in manifest['subsets']['development']['selection'])
    if set(dev['summary']['sequence_ids'])!=expected:raise ValueError('Development identities mismatch')
    shift=[]
    for j,c in enumerate(FEATURES):
        a=train['x'][:,j];b=dev['x'][:,j]
        a=a[np.isfinite(a)];b=b[np.isfinite(b)]
        ks=float(ks_2samp(a,b,method='asymp').statistic)
        scale=max(float(np.std(a,dtype=np.float64)),1e-8)
        shift.append({'feature':c,'ks_distance':ks,'mean_difference_in_train_std':float((np.mean(b,dtype=np.float64)-np.mean(a,dtype=np.float64))/scale),
                      'train_mean':float(np.mean(a,dtype=np.float64)),'development_mean':float(np.mean(b,dtype=np.float64))})
    shift.sort(key=lambda r:r['ks_distance'],reverse=True)
    csv_table(output/'feature_shift.csv',shift)
    xx=np.where(np.isfinite(train['x']),train['x'],0)
    matrix=np.corrcoef(xx,rowvar=False)
    csv_table(output/'train_feature_correlations.csv',[{'feature':c,**{d:float(matrix[j,k]) for k,d in enumerate(FEATURES)}} for j,c in enumerate(FEATURES)])
    scale=StandardScaler().fit(xx)
    pca=PCA(n_components=112,svd_solver='full').fit(scale.transform(xx))
    print('Fitting sequence-held-out diagnostic classifiers...',flush=True)
    # Prefix group IDs to prevent incidental numeric ID collisions across datasets.
    domain_x=np.r_[train['x'],dev['x']]
    domain_y=np.r_[np.zeros(len(train['x']),np.int8),np.ones(len(dev['x']),np.int8)]
    domain_groups=np.r_[train['groups'].astype(str),np.char.add('dev_',dev['groups'].astype(str))]
    domain=classifier_diagnostic(domain_x,domain_y,domain_groups,SEED)
    mask=classifier_diagnostic(dev['x'],dev['mask'].astype(np.int8),dev['groups'],SEED)
    summary={
        'version':1,'seed':SEED,'python':platform.python_version(),'metadata':metadata,
        'training':train['summary'],'development':dev['summary'],
        'domain_classification':domain,'scoring_mask_classification':mask,
        'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
        'largest_feature_shifts':shift[:20],
        'confirmation_or_reserved_rows_read':False,
        'validation_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sampling_note':'Full 20k rows for univariate/sequence summaries. Multivariate/KS/PCA/classifiers/feature-target correlations use 512 uniformly sampled rows per sequence from steps 1024..19999 to support causal lag comparisons. No IID inference.',
    }
    images=plot_all(train,dev,shift,pca.explained_variance_ratio_,output)
    summary['elapsed_seconds']=time.perf_counter()-started
    write_json(output/'summary.json',summary)
    build_html(output,summary,train,dev,shift,images)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pyarrow','matplotlib','scipy','scikit-learn','onnxruntime']}
    (output/'environment.txt').write_text('\n'.join(f'{name}=={v}' for name,v in versions.items())+'\n')
    print(json.dumps({'output':str(output),'elapsed_seconds':summary['elapsed_seconds'],
                      'domain':domain,'mask':mask,'top_shift':shift[:3],
                      'train_nonfinite':summary['training']['feature_nonfinite_count'],
                      'dev_nonfinite':summary['development']['feature_nonfinite_count']},indent=2),flush=True)


if __name__=='__main__':main()
