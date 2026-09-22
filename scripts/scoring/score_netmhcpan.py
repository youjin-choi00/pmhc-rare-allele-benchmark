#!/usr/bin/env python3
import pandas as pd, numpy as np, subprocess, tempfile, os, argparse, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / 'data/processed'
BIN  = {'4.1': BASE / 'tools/netMHCpan/netMHCpan-4.1/netMHCpan-4.1/netMHCpan',
        '4.2': BASE / 'tools/netMHCpan/netMHCpan-4.2/netMHCpan'}


def to_netmhcpan(allele):
    return allele.replace('*', '')


def score_allele(binary, allele, peptides):
    with tempfile.NamedTemporaryFile('w', suffix='.pep', delete=False) as f:
        f.write('\n'.join(peptides) + '\n')
        path = f.name
    try:
        r = subprocess.run([str(binary), '-p', path, '-a', to_netmhcpan(allele), '-BA'],
                           capture_output=True, text=True, timeout=3600)
    finally:
        os.unlink(path)

    out, wanted = {}, set(peptides)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[2] not in wanted:
            continue
        try:
            el_score  = float(parts[11])   # EL score
            affinity  = float(parts[15])   # Aff(nM)
        except (ValueError, IndexError):
            continue
        out[parts[2]] = (el_score, affinity)
    if not out:
        print(f'    WARNING {allele}: no rows parsed', flush=True)
        print('    stderr:', r.stderr[:200], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, choices=['4.1', '4.2'])
    ap.add_argument('--data', default='cv_folds.parquet',
                    help='input parquet in data/processed')
    ap.add_argument('--suffix', default='',
                    help='suffix for the output filename, e.g. _easyneg')
    args = ap.parse_args()
    binary = BIN[args.version]
    assert binary.exists(), f'not found: {binary}'

    b = pd.read_parquet(OUT / args.data)
    alleles = sorted(b.allele_norm.unique())
    print(f'{len(b):,} records, {len(alleles)} alleles, netMHCpan {args.version}')

    el, ba = {}, {}
    for i, al in enumerate(alleles, 1):
        peps = sorted(b.loc[b.allele_norm.eq(al), 'peptide'].unique())
        res = score_allele(binary, al, peps)
        for p, (e, a) in res.items():
            el[(al, p)] = e
            ba[(al, p)] = a
        print(f'  [{i:>3}/{len(alleles)}] {al:14s} {len(peps):>6,} peptides  {len(res):>6,} scored', flush=True)

    key = list(zip(b.allele_norm, b.peptide))
    b[f'nm{args.version}_el'] = [el.get(k, np.nan) for k in key]
    b[f'nm{args.version}_ba'] = [-np.log(max(ba.get(k, np.nan), 1e-9)) if k in ba else np.nan for k in key]

    miss = b[f'nm{args.version}_el'].isna().sum()
    print(f'\nunscored records: {miss:,} ({100*miss/len(b):.2f}%)')

    for col in [f'nm{args.version}_el', f'nm{args.version}_ba']:
        s = b.dropna(subset=[col])
        d = s[s.y][col].mean() - s[~s.y][col].mean()
        print(f'  {col}: mean(binder) - mean(non-binder) = {d:+.3f}  {"OK" if d > 0 else "*** INVERTED ***"}')

    path = OUT / f'scores_netmhcpan_{args.version}{args.suffix}.parquet'
    b[['allele_norm', 'peptide', f'nm{args.version}_el', f'nm{args.version}_ba']].to_parquet(path)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
