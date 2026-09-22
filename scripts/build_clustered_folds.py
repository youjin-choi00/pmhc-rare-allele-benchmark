#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1] / 'data/processed'
SEED, NFOLD = 42, 5


class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def components(peptides, max_dist: int):
    dsu = DSU(peptides)
    if max_dist >= 1:
        buckets = defaultdict(list)
        for p in peptides:
            for i in range(len(p)):
                buckets[(len(p), i, p[:i] + '*' + p[i + 1:])].append(p)
        for group in buckets.values():
            for q in group[1:]:
                dsu.union(group[0], q)
    if max_dist >= 2:
        buckets = defaultdict(list)
        for p in peptides:
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    key = (len(p), i, j,
                           p[:i] + '*' + p[i + 1:j] + '*' + p[j + 1:])
                    buckets[key].append(p)
        for group in buckets.values():
            for q in group[1:]:
                dsu.union(group[0], q)
    out = defaultdict(list)
    for p in peptides:
        out[dsu.find(p)].append(p)
    return list(out.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_dist', type=int, default=1, choices=[1, 2])
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--out', default='cv_folds_clustered.parquet')
    args = ap.parse_args()

    b = pd.read_parquet(B / 'cv_folds.parquet')
    peps = sorted(b.peptide.unique())
    print(f'{len(peps):,} distinct peptides')

    comps = components(peps, args.max_dist)
    sizes = np.array(sorted((len(c) for c in comps), reverse=True))
    print(f'{len(comps):,} components at edit distance <= {args.max_dist}')
    print(f'  singletons {int((sizes == 1).sum()):,}, largest {sizes[0]:,} '
          f'({100*sizes[0]/len(peps):.2f}% of peptides)')
    print(f'  top ten sizes: {list(sizes[:10])}')

    n_rec = b.groupby('peptide').size()
    weight = {i: int(sum(n_rec.get(p, 0) for p in c)) for i, c in enumerate(comps)}
    total = sum(weight.values())
    biggest = max(weight.values())
    print(f'\nrecords: {total:,}; largest component carries {biggest:,} '
          f'({100*biggest/total:.2f}%)')
    if biggest > total / NFOLD:
        print('  WARNING: one component exceeds a fold\'s share; folds cannot be balanced')

    order = sorted(weight, key=lambda i: -weight[i])
    load = [0] * NFOLD
    assign = {}
    for i in order:
        f = int(np.argmin(load))
        assign[i] = f
        load[f] += weight[i]
    pep_fold = {}
    for i, c in enumerate(comps):
        for p in c:
            pep_fold[p] = assign[i]

    nb = b.copy()
    nb['fold'] = nb.peptide.map(pep_fold)
    print('\nresulting folds:')
    t = nb.groupby('fold').agg(records=('y', 'size'), alleles=('allele_norm', 'nunique'),
                               binders=('y', 'sum'))
    print(t.to_string())
    ev = nb[nb.is_evaluable]
    g = ev.groupby('allele_norm').y.agg(['sum', 'size'])
    ok = g[(g['sum'] >= 10) & ((g['size'] - g['sum']) >= 10)]
    rare = ev.groupby('allele_norm').is_rare.first()
    print(f'\nevaluable after pooling: {len(ok)} alleles, '
          f'{sum(bool(rare.get(a, False)) for a in ok.index)} rare '
          f'(previously 79 and 17)')

    cross = 0
    buckets = defaultdict(set)
    for p in peps:
        for i in range(len(p)):
            buckets[(len(p), i, p[:i] + '*' + p[i + 1:])].add(p)
    for group in buckets.values():
        if len(group) > 1 and len({pep_fold[x] for x in group}) > 1:
            cross += len(group)
    print(f'peptides with a distance-one neighbour in another fold: {cross}')

    if args.dry_run:
        print('\ndry run; nothing written')
        return
    nb.to_parquet(B / args.out)
    print(f'\nsaved {B / args.out}')
    print('retrain both arms against this file, then rescore with the same suffix')


if __name__ == '__main__':
    main()
