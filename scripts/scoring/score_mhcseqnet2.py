#!/usr/bin/env python3
import argparse
import pandas as pd, numpy as np, subprocess, os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / 'data/processed'
TOOL = BASE / 'tools/MHCSeqNet2'
PY   = Path(os.environ.get('MHCSEQNET_PYTHON', 'python'))
WORK = BASE / 'data/processed/_mhcseqnet2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cv_folds.parquet',
                    help='input parquet in data/processed')
    ap.add_argument('--suffix', default='',
                    help='suffix for the output filename, e.g. _easyneg')
    args = ap.parse_args()

    b = pd.read_parquet(OUT / args.data)
    print(f'{len(b):,} records, {b.allele_norm.nunique()} alleles')

    WORK.mkdir(parents=True, exist_ok=True)
    inp  = WORK / 'input.csv'
    outp = WORK / 'output.csv'
    tmp  = WORK / 'tmp.csv'
    for f in (outp, tmp):
        if f.exists():
            f.unlink()

    pairs = b[['allele_norm', 'peptide']].drop_duplicates()
    pairs.columns = ['Allele', 'Peptide']
    pairs.to_csv(inp)
    print(f'wrote {len(pairs):,} unique pairs to {inp}')

    cmd = [str(PY), 'mhctool.py',
           '--MODE', 'CSV',
           '--CSV_PATH', str(inp),
           '--PEPTIDE_COLUMN_NAME', 'Peptide',
           '--ALLELE_COLUMN_NAME', 'Allele',
           '--ALLELE_MAPPER_PATH', 'resources/allele_mapper',
           '--OUTPUT_DIRECTORY', str(outp),
           '--TEMP_FILE_PATH', str(tmp),
           '--USE_ENSEMBLE']
    print('running:', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=TOOL, text=True)
    if r.returncode != 0:
        raise SystemExit(f'mhctool.py exited {r.returncode}')

    res = pd.read_csv(outp)
    print(f'read {len(res):,} predictions, columns: {list(res.columns)}')

    key = {(a, p): s for a, p, s in zip(res.Allele, res.Peptide, res.Prediction)}
    b['ms_score'] = [key.get(k, np.nan) for k in zip(b.allele_norm, b.peptide)]

    miss = int(b.ms_score.isna().sum())
    print(f'unscored records: {miss:,} ({100*miss/len(b):.2f}%)')

    s = b.dropna(subset=['ms_score'])
    d = s[s.y].ms_score.mean() - s[~s.y].ms_score.mean()
    print(f'  ms_score: mean(binder) - mean(non-binder) = {d:+.3f}  '
          f'{"OK" if d > 0 else "*** INVERTED — see docstring ***"}')

    path = OUT / f'scores_mhcseqnet2{args.suffix}.parquet'
    b[['allele_norm', 'peptide', 'ms_score']].to_parquet(path)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
