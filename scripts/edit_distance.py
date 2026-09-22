#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse
from pathlib import Path
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

BASE = Path(__file__).resolve().parents[1]
OUT  = BASE / 'data/processed'
T    = BASE / 'tools'
MAXD = 4          # Levenshtein distances beyond this are pooled into the 4+ band


def min_distances(queries, references, cap=MAXD):
    refs = list(references)
    out = {}
    B = 2000
    for i in range(0, len(queries), B):
        chunk = queries[i:i+B]
        M = process.cdist(chunk, refs, scorer=Levenshtein.distance,
                          score_cutoff=cap, workers=-1)
        for q, row in zip(chunk, M):
            out[q] = int(row.min())
        print(f'    {min(i+B, len(queries)):,}/{len(queries):,}', flush=True)
    return out


def reference_peptides(arm):
    if arm == 'mhcflurry':
        d = pd.read_csv(T / 'MHCflurry/4/2.2.0/models_class1_presentation/models/affinity_predictor_train_data.csv.bz2')
        return set(d.peptide.astype(str))
    if arm == 'mhcseqnet2':
        a = pd.read_csv(T / 'MHCSeqNet2/resources/datasets/raw_datasets/HLA_classI_MS_dataset_011320.tsv', sep='\t')
        b = pd.read_csv(T / 'MHCSeqNet2/resources/datasets/raw_datasets/antigen_information_051821_rev1.tsv', sep='\t')
        return set(a.Peptide.astype(str)) | set(b.Peptide.astype(str))
    if arm == 'alphafold':
        f = [pd.read_csv(T / f'alphafold_finetune/datasets_alphafold_finetune/pmhc_finetune/combo_1and2_{x}.tsv', sep='\t')
             for x in ('train', 'valid')]
        return set(pd.concat(f).target_chainseq.str.split('/').str[1].astype(str))
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['mhcflurry', 'mhcseqnet2', 'alphafold', 'cv'])
    args = ap.parse_args()

    b = pd.read_parquet(OUT / 'cv_folds.parquet')
    peptides = sorted(b.peptide.unique())
    print(f'{len(b):,} records, {len(peptides):,} distinct peptides')

    if args.arm == 'cv':
        dist = {}
        for f in sorted(b.fold.unique()):
            q   = sorted(b.loc[b.fold.eq(f), 'peptide'].unique())
            ref = b.loc[b.fold.ne(f), 'peptide'].unique()
            print(f'  fold {f}: {len(q):,} vs {len(ref):,}', flush=True)
            dist.update(min_distances(q, ref))
    else:
        ref = reference_peptides(args.arm)
        print(f'  reference set: {len(ref):,} peptides', flush=True)
        dist = min_distances(peptides, ref)

    col = f'editdist_{args.arm}'
    b[col] = b.peptide.map(dist)
    b[col] = b[col].where(b[col] <= MAXD, MAXD + 1)

    print(f'\ndistribution of {col}:')
    g = b.groupby(col).agg(records=('y', 'size'), alleles=('allele_norm', 'nunique'), binders=('y', 'mean'))
    g.index = [str(int(i)) if i <= MAXD else f'{MAXD}+' for i in g.index]
    print(g.round(3).to_string())

    out = OUT / f'editdist_{args.arm}.parquet'
    b[['allele_norm', 'peptide', col]].drop_duplicates().to_parquet(out)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
