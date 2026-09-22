#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--experiment', default='ms_cv',
                help='experiment_name given to train.py')
ap.add_argument('--folds', default='cv_folds.parquet',
                help='the partition the model was trained against')
ap.add_argument('--suffix', default='',
                help='appended to the output filename')
args = ap.parse_args()

OUT  = Path(__file__).resolve().parents[2] / 'data/processed'
WORK = OUT / 'ms_retrain' / args.experiment
NF   = 5

parts = []
for n in range(1, NF + 1):
    p = pd.read_csv(WORK / f'{args.experiment}_{n}-5' / 'pred_test.csv',
                    index_col=0)
    p['cv_fold'] = n - 1
    print(f'fold {n-1}: {len(p):,} predictions')
    parts.append(p)
a = pd.concat(parts, ignore_index=True)

b = pd.read_parquet(OUT / args.folds)
m = b[['allele_norm', 'peptide', 'y', 'fold']].merge(
    a.rename(columns={'Allele': 'allele_norm', 'Peptide': 'peptide',
                      'Prediction': 'ms_cv_score'}),
    on=['allele_norm', 'peptide'], how='left')
print(f'\njoined {int(m.ms_cv_score.notna().sum()):,} of {len(b):,}')
assert (m.dropna(subset=['cv_fold']).fold == m.dropna(subset=['cv_fold']).cv_fold).all(), \
    'fold mismatch: a record was predicted by the wrong fold model'
print('fold alignment: OK (every record predicted by the model that held it out)')

s = m.dropna(subset=['ms_cv_score'])
d = s[s.y].ms_cv_score.mean() - s[~s.y].ms_cv_score.mean()
print(f'ms_cv_score: mean(binder) - mean(non-binder) = {d:+.3f}  '
      f'{"OK" if d > 0 else "*** INVERTED ***"}')
print(f'  isGenerated agrees with y: {bool((s.isGenerated.astype(bool) == ~s.y).all())}')

m[['allele_norm', 'peptide', 'ms_cv_score']].to_parquet(OUT / f'scores_mhcseqnet2_cv{args.suffix}.parquet')
print(f'\nsaved {OUT}/scores_mhcseqnet2_cv{args.suffix}.parquet')
