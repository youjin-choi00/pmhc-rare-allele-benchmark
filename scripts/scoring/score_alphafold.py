#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parents[2] / 'data/processed'
P   = 'model_2_ptm_ft_'

t = pd.read_csv(OUT / 'af_scores_final.tsv', sep='\t')
print(f'{len(t):,} predictions, {t.targetid.str.split("|").str[0].nunique()} alleles')

t[['allele_norm', 'peptide']] = t.targetid.str.split('|', n=1, expand=True)
t['ipae'] = t[[P + 'pae_0_1', P + 'pae_1_0']].mean(axis=1)

b = pd.read_parquet(OUT / 'cv_folds.parquet')
m = t.merge(b[['allele_norm', 'peptide', 'y', 'fold', 'is_rare', 'is_evaluable']],
            on=['allele_norm', 'peptide'], how='inner')
print(f'joined {len(m):,} of {len(t):,}   binders {int(m.y.sum()):,}  '
      f'non-binders {int((~m.y).sum()):,} ({100*m.y.mean():.1f}% positive)')
if len(m) != len(t):
    raise SystemExit(f'join lost {len(t)-len(m)} rows; expected one-to-one')

cands = [('peptide pLDDT',        P + 'plddt_1',  +1),
         ('complex pLDDT',        P + 'plddt',    +1),
         ('MHC pLDDT',            P + 'plddt_0',  +1),
         ('PAE MHC->peptide',     P + 'pae_0_1',  -1),
         ('PAE peptide->MHC',     P + 'pae_1_0',  -1),
         ('interface PAE (mean)', 'ipae',         -1),
         ('PAE peptide-internal', P + 'pae_1_1',  -1),
         ('complex PAE',          P + 'pae',      -1)]

print(f'\n{"quantity":24s} {"AUC":>7} {"range":>18}')
for name, col, sign in cands:
    auc = roc_auc_score(m.y, sign * m[col])
    print(f'{name:24s} {auc:7.4f} {m[col].min():8.1f} -{m[col].max():8.1f}')

m['af_ipae'] = -m.ipae
diff = m[m.y].af_ipae.mean() - m[~m.y].af_ipae.mean()
print(f'\naf_ipae: mean(binder) - mean(non-binder) = {diff:+.3f}  '
      f'{"OK" if diff > 0 else "*** INVERTED ***"}')

keep = ['allele_norm', 'peptide', 'af_ipae',
        P + 'plddt_1', P + 'pae_0_1', P + 'pae_1_0', P + 'plddt']
m[keep].to_parquet(OUT / 'scores_alphafold.parquet')
print(f'saved {OUT}/scores_alphafold.parquet')
