#!/usr/bin/env python3
import argparse
import pandas as pd, numpy as np
from pathlib import Path
from mhcflurry import Class1AffinityPredictor

BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / 'data/processed'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cv_folds.parquet',
                    help='input parquet in data/processed')
    ap.add_argument('--suffix', default='',
                    help='suffix for the output filename, e.g. _easyneg')
    args = ap.parse_args()

    b = pd.read_parquet(OUT / args.data)
    print(f'{len(b):,} records, {b.allele_norm.nunique()} alleles')

    aff = Class1AffinityPredictor.load()
    print(f'affinity predictor: {len(aff.supported_alleles):,} supported alleles')

    supported = set(aff.supported_alleles)
    unsupported = sorted(set(b.allele_norm) - supported)
    if unsupported:
        print(f'unsupported alleles ({len(unsupported)}): {unsupported[:10]}')

    m = b.allele_norm.isin(supported)
    print(f'scoring {int(m.sum()):,} records ({int((~m).sum()):,} unsupported)')

    sub = b[m]
    nm = aff.predict(alleles=sub.allele_norm.tolist(), peptides=sub.peptide.tolist())
    b.loc[m, 'mf_ba'] = -np.log(np.clip(nm, 1e-9, None))

    for col in ['mf_ba']:
        s = b.dropna(subset=[col])
        d = s[s.y][col].mean() - s[~s.y][col].mean()
        print(f'  {col}: mean(binder) - mean(non-binder) = {d:+.3f}  {"OK" if d > 0 else "*** INVERTED ***"}')
        print(f'    missing: {b[col].isna().sum():,}')

    path = OUT / f'scores_mhcflurry{args.suffix}.parquet'
    b[['allele_norm', 'peptide', 'mf_ba']].to_parquet(path)
    print(f'\nsaved {path}')

if __name__ == '__main__':
    main()
