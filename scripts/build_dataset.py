#!/usr/bin/env python3
import pandas as pd, numpy as np, re
from pathlib import Path

BASE   = Path(__file__).resolve().parents[1]
RAW    = BASE / 'data/raw/iedb_subset.parquet'
OUT    = BASE / 'data/processed'
SEED   = 42
NFOLDS = 5

AA          = set('ACDEFGHIKLMNPQRSTVWY')
ALLELE_RE   = r'^HLA-[ABCEG]\*\d+:\d+[NLSCAQ]?$'
BINDER_LABS = ['Positive-High', 'Positive-Intermediate']
NEG_LAB     = 'Negative'
RARE_MAX    = 200          # Section 3.6
MIN_PER_CLS = 10           # Minimum records of each class for an allele to be evaluable


def normalize_allele(a):
    a = str(a).strip().upper().replace('HLA-', '').replace('*', '').replace(':', '')
    m = re.match(r'^([ABCEG])(\d{2})(\d{2,})([NLSCAQ]?)$', a)
    return f'HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}{m.group(4)}' if m else None


def evaluable(df, min_per_class=MIN_PER_CLS):
    pos = df[df.y].allele_norm.value_counts()
    neg = df[~df.y].allele_norm.value_counts()
    return sorted(set(pos[pos >= min_per_class].index) & set(neg[neg >= min_per_class].index))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(RAW)

    log = [('all MHC ligand records', d)]
    x = d[d.mhc_class.eq('I')];                                    log.append(('MHC class I', x))
    x = x[x.allele.astype(str).str.startswith('HLA-')];             log.append(('human HLA', x))
    x = x.copy()
    x['L'] = x.peptide.astype(str).str.len()
    x = x[x.L.between(8, 11)];                                      log.append(('peptide length 8-11', x))
    x = x[x.peptide.astype(str).apply(lambda p: set(p) <= AA)];      log.append(('standard amino acids', x))
    x = x[x.allele.astype(str).str.match(ALLELE_RE)];               log.append(('two-field allele name', x))

    print(f'{"step":26s} {"records":>10} {"peptides":>10} {"alleles":>8}')
    rows = []
    for tag, t in log:
        print(f'{tag:26s} {len(t):>10,} {t.peptide.nunique():>10,} {t.allele.nunique():>8,}')
        rows.append(dict(step=tag, records=len(t), peptides=t.peptide.nunique(), alleles=t.allele.nunique()))
    pd.DataFrame(rows).to_csv(OUT / 'filter_log.csv', index=False)

    x = x.copy()
    x['allele_norm'] = x.allele.map(normalize_allele)
    method = x.method.astype(str)
    x['paradigm'] = 'other'
    x.loc[method.str.contains('mass spectrometry', case=False), 'paradigm'] = 'elution'
    x.loc[method.str.contains('competitive|direct', case=False)
          | method.isin(['purified MHC', 'binding assay']), 'paradigm'] = 'binding'

    print('\nparadigm split')
    print(x.paradigm.value_counts().to_string())
    x.to_parquet(OUT / 'iedb_filtered.parquet')

    binding = x[x.paradigm.eq('binding')].copy()
    binding.to_parquet(OUT / 'binding_all_lengths.parquet')
    binding[binding.L.eq(9)].to_parquet(OUT / 'binding_9mer.parquet')

    pair = ['allele_norm', 'peptide']
    CONFLICT_POS = BINDER_LABS + ['Positive']
    has_neg = binding.qualitative.eq(NEG_LAB).groupby([binding[c] for c in pair]).any()
    has_pos = binding.qualitative.isin(CONFLICT_POS).groupby([binding[c] for c in pair]).any()
    conflicting = has_neg[has_neg & has_pos].index
    print(f'\nrepeated measurements:')
    print(f'  conflicting pairs dropped  : {len(conflicting):,}')

    binding = binding[~binding.set_index(pair).index.isin(conflicting)]

    b = binding[binding.qualitative.isin(BINDER_LABS + [NEG_LAB])].copy()
    b['y'] = b.qualitative.ne(NEG_LAB)
    print(f'\nlabel definition: {len(b):,} of {len(binding):,} records retained '
          f'({100*len(b)/len(binding):.0f}%), {b.allele_norm.nunique()} alleles')
    print(f'  binders {int(b.y.sum()):,}  non-binders {int((~b.y).sum()):,} '
          f'({100*b.y.mean():.1f}% positive)')

    agg = {c: 'first' for c in b.columns if c not in pair}
    if 'quantitative' in b.columns:
        agg['quantitative'] = lambda x: pd.to_numeric(x, errors='coerce').median()
    isnm = b.units.eq('nM')
    qnm  = pd.to_numeric(b.quantitative, errors='coerce').where(isnm)
    inq  = b.inequality.fillna('=').replace({'>=': '>'}).where(isnm)
    nmagg = (b.assign(_q=qnm, _i=inq)
               .groupby(pair, as_index=False)
               .agg(affinity_nm=('_q', 'median'),
                    inequality_nm=('_i', lambda s: '=' if (s == '=').any()
                                   else ('>' if s.notna().any() else None))))

    n_before = len(b)
    b = b.groupby(pair, as_index=False).agg(agg)
    b = b.merge(nmagg, on=pair, how='left')
    print(f'  duplicate records collapsed: {n_before - len(b):,}')
    print(f'  pairs with a nanomolar affinity: {int(b.affinity_nm.notna().sum()):,}')
    print(f'  remaining: {len(b):,} records, one per pair')

    b.to_parquet(OUT / 'binding_labelled.parquet')
    print(f"\nsaved the labelled, deduplicated set: {len(b):,} records, "
          f"{b.allele_norm.nunique()} alleles")

    n_before = len(b)
    a_before = b.allele_norm.nunique()
    b = b[b.L.eq(9)].copy()
    print(f'\nrestricted to nine-mers: {len(b):,} of {n_before:,} records '
          f'({b.allele_norm.nunique()} of {a_before} alleles)')

    rng = np.random.default_rng(SEED)
    peptides = np.array(sorted(b.peptide.unique()))
    b['fold'] = b.peptide.map(pd.Series(rng.permutation(len(peptides)) % NFOLDS, index=peptides))

    print(f'\n{NFOLDS}-fold partition by peptide (seed {SEED})')
    print(b.groupby('fold').agg(records=('y', 'size'),
                                alleles=('allele_norm', 'nunique'),
                                binders=('y', 'sum')).to_string())

    depth = (binding.drop_duplicates(['allele_norm', 'peptide'])
             .allele_norm.value_counts())          # Distinct pairs, before label filtering
    u = evaluable(b)
    rare = [a for a in u if depth.get(a, 0) < RARE_MAX]
    print(f'\nevaluable alleles (>={MIN_PER_CLS} of each class): {len(u)}')
    print(f'  rare (<{RARE_MAX} records): {len(rare)}   common: {len(u)-len(rare)}')

    b['is_rare']     = b.allele_norm.map(lambda a: depth.get(a, 0) < RARE_MAX)
    b['n_records']   = b.allele_norm.map(depth)
    b['is_evaluable'] = b.allele_norm.isin(u)
    b.to_parquet(OUT / 'cv_folds.parquet')

    print(f'\nwritten to {OUT}/')
    for f in ['filter_log.csv', 'iedb_filtered.parquet', 'binding_all_lengths.parquet',
              'binding_9mer.parquet', 'cv_folds.parquet']:
        print(f'  {f}')


if __name__ == '__main__':
    main()
