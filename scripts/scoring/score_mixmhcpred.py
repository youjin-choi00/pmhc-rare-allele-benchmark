#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT  = BASE / 'data/processed'
TOOL = BASE / 'tools/MixMHCpred'
EXE  = TOOL / 'MixMHCpred'


def to_tool(allele: str) -> str:
    return allele.replace('HLA-', '').replace('*', '').replace(':', '')


def supported() -> set[str]:
    out = set()
    for line in open(TOOL / 'lib/alleles_list.txt'):
        line = line.strip()
        if not line or line.startswith('Allele'):
            continue
        out.add(line.split()[0])
    return out


def score_allele(peptides: list[str], allele: str, workdir: Path):
    inp = workdir / 'pep.txt'
    outp = workdir / 'out.txt'
    inp.write_text('\n'.join(peptides) + '\n')
    if outp.exists():
        outp.unlink()
    r = subprocess.run([str(EXE), '-i', str(inp), '-o', str(outp), '-a', allele],
                       cwd=TOOL, text=True, capture_output=True)
    if r.returncode != 0 or not outp.exists():
        print(f'  {allele}: failed ({r.returncode}) {r.stderr.strip()[:120]}', file=sys.stderr)
        return None
    rows = [l for l in outp.read_text().splitlines() if l and not l.startswith('#')]
    if len(rows) < 2:
        return None
    df = pd.read_csv(pd.io.common.StringIO('\n'.join(rows)), sep='\t')
    return df[['Peptide', 'Score_bestAllele', '%Rank_bestAllele']]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cv_folds.parquet',
                    help='input parquet in data/processed')
    ap.add_argument('--suffix', default='',
                    help='suffix for the output filename, e.g. _easyneg')
    args = ap.parse_args()

    b = pd.read_parquet(OUT / args.data)
    sup = supported()
    alleles = sorted(b.allele_norm.unique())
    usable = [a for a in alleles if to_tool(a) in sup]
    print(f'{len(b):,} records, {len(alleles)} alleles, '
          f'{len(usable)} supported by MixMHCpred')
    missing = [a for a in alleles if a not in usable]
    if missing:
        print(f'  not supported: {missing}')

    score, rank = {}, {}
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for i, allele in enumerate(usable, 1):
            peps = sorted(b.loc[b.allele_norm.eq(allele), 'peptide'].unique())
            res = score_allele(list(peps), to_tool(allele), wd)
            n = 0 if res is None else len(res)
            print(f'  [{i:3d}/{len(usable)}] {allele:15s} {len(peps):7,} peptides '
                  f'{n:7,} scored', flush=True)
            if res is None:
                continue
            for p, s, rk in zip(res.Peptide, res.Score_bestAllele, res['%Rank_bestAllele']):
                score[(allele, p)] = s
                rank[(allele, p)] = rk

    key = list(zip(b.allele_norm, b.peptide))
    b['mix_score'] = [score.get(k, np.nan) for k in key]
    b['mix_rank'] = [rank.get(k, np.nan) for k in key]

    miss = int(b.mix_score.isna().sum())
    print(f'\nunscored records: {miss:,} ({100 * miss / len(b):.2f}%)')
    s = b.dropna(subset=['mix_score'])
    for col, sign in (('mix_score', +1), ('mix_rank', -1)):
        d = s[s.y][col].mean() - s[~s.y][col].mean()
        ok = 'OK' if sign * d > 0 else '*** INVERTED ***'
        print(f'  {col}: mean(binder) - mean(non-binder) = {d:+.4f}  '
              f'(expected {"positive" if sign > 0 else "negative"})  {ok}')

    path = OUT / f'scores_mixmhcpred{args.suffix}.parquet'
    b[['allele_norm', 'peptide', 'mix_score', 'mix_rank']].to_parquet(path)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()