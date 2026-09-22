#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

BASE    = Path(__file__).resolve().parents[1] / 'data/processed'
KEY     = ['allele_norm', 'peptide']
SEED    = 42
NBOOT   = 2000
MIN_CLS = 10     # Per-allele minimum of each class, rank metrics
MIN_AFF = 20     # Per-allele minimum of exactly-measured records, correlation

NO_MASK_REASON = {
    'ESMFold, not pMHC-adapted': 'not applicable',
    'MHCflurry 2.0, retrained': 'not required',
    'MHCSeqNet2, retrained':    'not required',
    'MHCSeqNet2, IEDB, generated negatives': 'not required',
}

ARMS = [
    ('NetMHCpan 4.1, pre-trained',  'nm4.1_el',    'nm4.1_ba',    'editdist_netmhcpan'),
    ('NetMHCpan 4.2, pre-trained',  'nm4.2_el',    'nm4.2_ba',    'editdist_netmhcpan_42'),
    ('MHCflurry 2.0, pre-trained',    'mf_ba',       'mf_ba',       'editdist_mhcflurry'),
    ('MHCflurry 2.0, retrained',      'mf_cv_ba',    'mf_cv_ba',    None),
    ('MHCSeqNet2, pre-trained',       'ms_score',    'ms_score',    'editdist_mhcseqnet2'),
    ('MHCSeqNet2, retrained',         'ms_cv_score', 'ms_cv_score', None),
    ('MixMHCpred 3.0, pre-trained', 'mix_score',   'mix_score',   'editdist_mixmhcpred'),
    ('AlphaFold, fine-tuned',         'af_cpae',     'af_cpae',     'editdist_alphafold'),
    ('AlphaFold, geometric features', 'af_geom',     'af_geom',     'editdist_alphafold'),
    ('ESMFold, not pMHC-adapted',    'esm_ipae',    'esm_ipae',    None),
    ('MHC-Fine, structure-adapted',  'mf_ipae',     'mf_ipae',     'editdist_mhcfine'),
    ('MHCSeqNet2, IEDB, generated negatives',  'ms_gen_score',   'ms_gen_score',   None),
    ('MHCSeqNet2, Atlas',                      'ms_atlas_score', 'ms_atlas_score', 'in_atlas'),
    ('MHCSeqNet2, Atlas and IEDB',             'ms_both_score',  'ms_both_score',  'in_atlas'),
]

METRICS = ['roc_auc', 'pr_auc', 'r_precision', 'spearman']


def load() -> pd.DataFrame:
    d = pd.read_parquet(BASE / 'cv_folds.parquet')[
        KEY + ['y', 'is_rare', 'n_records', 'affinity_nm', 'inequality_nm']]
    for f in ('editdist_mhcflurry', 'editdist_mhcseqnet2', 'editdist_alphafold',
              'editdist_mixmhcpred', 'editdist_netmhcpan', 'editdist_mhcfine'):
        d = d.merge(pd.read_parquet(BASE / f'{f}.parquet'), on=KEY)
    d = d.merge(pd.read_parquet(BASE / 'editdist_netmhcpan_42.parquet'), on=KEY)

    af = pd.read_csv(BASE / 'af_scores_final.tsv', sep='\t')
    af[['allele_norm', 'peptide']] = af.targetid.str.split('|', n=1, expand=True)
    af['af_cpae'] = -af['model_2_ptm_ft_pae']
    d = d.merge(af[KEY + ['af_cpae']], on=KEY, how='left')

    geom = BASE / 'scores_alphafold_geom.parquet'
    if geom.exists():
        d = d.merge(pd.read_parquet(geom)[KEY + ['af_geom']], on=KEY, how='left')
    else:
        d['af_geom'] = np.nan
        print('WARNING: scores_alphafold_geom.parquet not found', file=sys.stderr)

    for f, cols in [('scores_netmhcpan_4.1', ['nm4.1_el', 'nm4.1_ba']),
                    ('scores_netmhcpan_4.2', ['nm4.2_el', 'nm4.2_ba']),
                    ('scores_mhcflurry',     ['mf_ba']),
                    ('scores_mhcflurry_cv_clustered',  ['mf_cv_ba']),
                    ('scores_mhcseqnet2',    ['ms_score']),
                    ('scores_mhcseqnet2_cv_clustered', ['ms_cv_score']),
                    ('scores_mixmhcpred',    ['mix_score', 'mix_rank']),
                    ('scores_esmfold',       ['esm_ipae']),
                    ]:
        d = d.merge(pd.read_parquet(BASE / f'{f}.parquet')[KEY + cols], on=KEY, how='left')
    for f, col in (('scores_mhcseqnet2_cv_iedb_gen',   'ms_gen_score'),
                   ('scores_mhcseqnet2_cv_atlas',      'ms_atlas_score'),
                   ('scores_mhcseqnet2_cv_atlas_iedb', 'ms_both_score')):
        q = BASE / f'{f}.parquet'
        if q.exists():
            t = pd.read_parquet(q)
            sc = [c for c in t.columns if c not in KEY][0]
            d = d.merge(t[KEY + [sc]].rename(columns={sc: col}), on=KEY, how='left')
        else:
            d[col] = np.nan
            print(f'WARNING: {f}.parquet not found', file=sys.stderr)

    lig = BASE.parent / 'raw/mhcmotifatlas_ligands.txt'
    if lig.exists():
        a = pd.read_csv(lig, sep=r'\s+')
        a.columns = [c.strip() for c in a.columns]

        def _norm(x):
            m = re.match(r'^([ABC])(\d{2})(\d{2})$', str(x).strip())
            return f'HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}' if m else None
        seen = set(zip(a.Allele.map(_norm), a.Peptide))
        d['in_atlas'] = [0 if (x, y) in seen else 1
                         for x, y in zip(d.allele_norm, d.peptide)]
    else:
        d['in_atlas'] = 1
        print('WARNING: mhcmotifatlas_ligands.txt not found; Atlas arms unexcluded',
              file=sys.stderr)

    mf = BASE / 'scores_mhcfine.parquet'
    if mf.exists():
        d = d.merge(pd.read_parquet(mf)[KEY + ['mf_ipae']], on=KEY, how='left')
    else:
        d['mf_ipae'] = np.nan
        print('WARNING: scores_mhcfine.parquet not found; MHC-Fine arm empty',
              file=sys.stderr)
    return d


def orient(d: pd.DataFrame, col: str) -> int:
    return 1 if d.loc[d.y, col].mean() > d.loc[~d.y, col].mean() else -1


def restrict(d: pd.DataFrame, col: str, mask: str | None) -> pd.DataFrame:
    d = d.dropna(subset=[col])
    return d[d[mask] >= 1] if mask else d


def r_precision(y: np.ndarray, s: np.ndarray):
    k = int(y.sum())
    return None if k == 0 or k > len(y) else float(y[np.argsort(-s)][:k].mean())


def per_allele(d: pd.DataFrame, col: str, metric: str,
               min_cls: int | None = None) -> pd.DataFrame:
    m_cls = MIN_CLS if min_cls is None else min_cls
    sign, rows = orient(d, col), []
    for allele, g in d.groupby('allele_norm'):
        pos, neg = int(g.y.sum()), int((~g.y).sum())
        s = (sign * g[col]).to_numpy()
        y = g.y.to_numpy()
        if metric == 'spearman':
            if len(g) < MIN_AFF or g.affinity_nm.nunique() < 2:
                continue
            v = spearmanr(s, -g.affinity_nm.to_numpy()).statistic
        else:
            if pos < m_cls or neg < m_cls:
                continue
            v = {'roc_auc':     lambda: roc_auc_score(y, s),
                 'pr_auc':      lambda: average_precision_score(y, s),
                 'r_precision': lambda: r_precision(y, s)}[metric]()
        if v is None or not np.isfinite(v):
            continue
        rows.append(dict(allele=allele, value=float(v), is_rare=bool(g.is_rare.iloc[0]),
                         depth=int(g.n_records.iloc[0]), n=len(g), pos=pos, neg=neg))
    return pd.DataFrame(rows)


def gap_with_ci(pa: pd.DataFrame, rng) -> dict:
    r = pa.loc[pa.is_rare, 'value'].to_numpy()
    c = pa.loc[~pa.is_rare, 'value'].to_numpy()
    out = dict(n_alleles=len(pa), n_rare=len(r), n_common=len(c),
               all=float(np.median(pa.value)) if len(pa) else np.nan,
               common=float(np.median(c)) if len(c) else np.nan,
               rare=float(np.median(r)) if len(r) else np.nan,
               gap=np.nan, lo=np.nan, hi=np.nan)
    if len(r) >= 2 and len(c) >= 2:
        boot = [np.median(rng.choice(c, len(c))) - np.median(rng.choice(r, len(r)))
                for _ in range(NBOOT)]
        out.update(gap=out['common'] - out['rare'],
                   lo=float(np.percentile(boot, 2.5)),
                   hi=float(np.percentile(boot, 97.5)))
    return out


def depth_trend(pa: pd.DataFrame, rng) -> dict:
    if len(pa) < 6:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, n_alleles=len(pa))
    v, n = pa.value.to_numpy(), np.log10(pa.depth.to_numpy())
    boot = [spearmanr(v[i], n[i]).statistic
            for i in (rng.integers(0, len(v), len(v)) for _ in range(NBOOT))]
    boot = np.array([x for x in boot if np.isfinite(x)])
    return dict(rho=float(spearmanr(v, n).statistic), n_alleles=len(pa),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def collect(d: pd.DataFrame, scope: str):
    summary, long = [], []
    exact = d[d.affinity_nm.notna() & d.inequality_nm.eq('=')]
    for name, rank_col, aff_col, mask in ARMS:
        for metric in METRICS:
            src = exact if metric == 'spearman' else d
            col = aff_col if metric == 'spearman' else rank_col
            sub = restrict(src, col, mask)
            base = dict(scope=scope, arm=name, metric=metric, n_records=len(sub))
            if len(sub) < 100:
                summary.append({**base, 'note': 'insufficient records'})
                continue
            pa = per_allele(sub, col, metric)
            if len(pa) < 4:
                summary.append({**base, 'note': f'{len(pa)} alleles',
                                'n_alleles': len(pa),
                                'n_rare': int(pa.is_rare.sum()) if len(pa) else 0})
                continue
            row = {**base, **gap_with_ci(pa, np.random.default_rng(SEED))}
            if metric == 'roc_auc':
                row.update({f'depth_{k}': v for k, v in
                            depth_trend(pa, np.random.default_rng(SEED)).items()})
            summary.append(row)
            long.append(pa.assign(scope=scope, arm=name, metric=metric))
    return pd.DataFrame(summary), (pd.concat(long, ignore_index=True) if long else pd.DataFrame())


MINUS = '\u2212'


def num(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return (f'{x:+.4f}' if sign else f'{x:.4f}').replace('-', MINUS)


def ci(lo, hi) -> str:
    if lo is None or not np.isfinite(lo):
        return '—'
    return f'[{lo:+.4f}, {hi:+.4f}]'.replace('-', MINUS)


def group_table(summary: pd.DataFrame, scope: str, metric: str, caption: str) -> str:
    rows = summary[(summary.scope == scope) & (summary.metric == metric)]
    out = [caption, '',
           '| Predictor | Records | All | Common | Rare | Gap | 95% CI | Alleles (rare) |',
           '|:--|--:|--:|--:|--:|--:|:--|:--|']
    for _, r in rows.iterrows():
        if isinstance(r.get('note'), str):
            alleles = ('—' if pd.isna(r.get('n_alleles'))
                       else f'{int(r.n_alleles)} ({int(r.n_rare)})')
            out.append(f'| {r.arm} | {int(r.n_records):,} | — | — | — | not estimable | — | '
                       f'{alleles} |')
            continue
        out.append(f'| {r.arm} | {int(r.n_records):,} | {num(r["all"])} | {num(r["common"])} | '
                   f'{num(r["rare"])} | {num(r["gap"], sign=True)} | {ci(r["lo"], r["hi"])} | '
                   f'{int(r.n_alleles)} ({int(r.n_rare)}) |')
    return '\n'.join(out) + '\n'


def depth_table(summary: pd.DataFrame, scope: str) -> str:
    rows = summary[(summary.scope == scope) & (summary.metric == 'roc_auc')]
    rows = rows[rows['depth_rho'].notna()] if 'depth_rho' in rows else rows.iloc[0:0]
    out = ['**Study depth as a continuous covariate.** Spearman correlation between per-allele '
           "AUC-ROC and the base-ten logarithm of the allele's record count. A positive value "
           'indicates better prediction for better-studied alleles.', '',
           '| Predictor | Alleles | rho | 95% CI |', '|:--|--:|--:|:--|']
    for _, r in rows.iterrows():
        out.append(f'| {r.arm} | {int(r.depth_n_alleles)} | {num(r.depth_rho, sign=True)} | '
                   f'{ci(r.depth_lo, r.depth_hi)} |')
    return '\n'.join(out) + '\n'


def leakage_table(d: pd.DataFrame) -> str:
    out = ["**Effect of excluding each predictor's own training peptides.** Median per-allele "
           'AUC-ROC, over alleles retaining at least ten records of each class.', '',
           '| Predictor | Without exclusion | With exclusion | Records after exclusion | Change |',
           '|:--|--:|--:|--:|--:|']
    for name, col, _, mask in ARMS:
        base = d.dropna(subset=[col])
        if len(base) < 100:
            continue
        before = float(np.median(per_allele(base, col, 'roc_auc').value))
        if mask is None:
            out.append(f'| {name} | {num(before)} | {NO_MASK_REASON.get(name, "—")} | — | — |')
            continue
        sub = base[base[mask] >= 1]
        after = float(np.median(per_allele(sub, col, 'roc_auc').value))
        out.append(f'| {name} | {num(before)} | {num(after)} | {len(sub):,} | '
                   f'{num(after - before, sign=True)} |')
    return '\n'.join(out) + '\n'


def mechanism_table() -> str:
    p = BASE / 'af_geometry.parquet'
    if not p.exists():
        return '_af_geometry.parquet not found; mechanism table skipped._\n'
    g = pd.read_parquet(p)
    b = pd.read_parquet(BASE / 'cv_folds.parquet')[KEY + ['y', 'is_rare']]
    b = b.merge(pd.read_parquet(BASE / 'editdist_alphafold.parquet'), on=KEY)
    m = g.merge(b, on=KEY)
    out = ['**Peptides not placed in the groove.** Proportion of predictions with no groove '
           'heavy atom within four angstroms of any peptide atom.', '',
           '| Set | Group | Binders | Non-binders |', '|:--|:--|--:|--:|']
    for label, frame in (('all predictions', m),
                         ('leakage-controlled', m[m.editdist_alphafold >= 1])):
        for grp, sub in (('common', frame[~frame.is_rare]), ('rare', frame[frame.is_rare])):
            out.append(f'| {label} | {grp} | {sub[sub.y].contacts_4.eq(0).mean() * 100:.1f}% | '
                       f'{sub[~sub.y].contacts_4.eq(0).mean() * 100:.1f}% |')
    return '\n'.join(out) + '\n'


def paired_structural(d: pd.DataFrame) -> str:
    sub = d[d.af_cpae.notna() & d.af_geom.notna() & (d.editdist_alphafold >= 1)]
    if len(sub) < 100:
        return '_paired structural comparison unavailable._\n'
    per = {a: (roc_auc_score(g.y, g.af_cpae), roc_auc_score(g.y, g.af_geom))
           for a, g in sub.groupby('allele_norm')
           if g.y.sum() >= MIN_CLS and (~g.y).sum() >= MIN_CLS}
    A = np.array([v[0] for v in per.values()])
    G = np.array([v[1] for v in per.values()])
    rng = np.random.default_rng(SEED)
    boot = [np.median(G[i]) - np.median(A[i])
            for i in (rng.integers(0, len(A), len(A)) for _ in range(NBOOT))]
    return ("**Predicted geometry against the model's own confidence.** Paired by allele over "
            f'{len(sub):,} records and {len(A)} alleles. Difference in median per-allele '
            f'AUC-ROC {num(float(np.median(G) - np.median(A)), sign=True)}, 95% CI '
            f'{ci(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))}.\n')


def _fam(a):
    m = re.match(r'HLA-([ABC])\*(\d{2})', a)
    return f'{m.group(1)}{m.group(2)}' if m else a


def _gap(pa, summary=np.median):
    if not len(pa) or 'is_rare' not in pa.columns:
        return np.nan
    r = pa.loc[pa.is_rare, 'value'].to_numpy()
    c = pa.loc[~pa.is_rare, 'value'].to_numpy()
    return float(summary(c) - summary(r)) if len(r) >= 2 and len(c) >= 2 else np.nan


def _boot(pa, rng, by_family=False):
    v, r = pa.value.to_numpy(), pa.is_rare.to_numpy()
    if by_family:
        fam = np.array([_fam(a) for a in pa.allele])
        fams = np.array(sorted(set(fam)))
        idx = {f: np.where(fam == f)[0] for f in fams}
    out = []
    for _ in range(NBOOT):
        i = (np.concatenate([idx[f] for f in rng.choice(fams, len(fams))])
             if by_family else rng.integers(0, len(v), len(v)))
        if r[i].sum() < 2 or (~r[i]).sum() < 2:
            continue
        out.append(np.median(v[i][~r[i]]) - np.median(v[i][r[i]]))
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) >= 100 else (np.nan, np.nan)


def _tab(header, rows):
    sep = '|:--' + '|--:' * (len(header) - 1) + '|'
    body = ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(['| ' + ' | '.join(header) + ' |', sep] + body) + '\n'


def robustness_section(d):
    rng = np.random.default_rng(SEED)
    out = []

    rows = []
    for name, col, _, mask in ARMS:
        pa = per_allele(restrict(d, col, mask), col, 'roc_auc')
        if len(pa) < 8 or 'is_rare' not in pa.columns or pa.is_rare.sum() < 2:
            rows.append([name, len(pa), '-', '-', '-', '-']); continue
        a = _boot(pa, np.random.default_rng(SEED))
        f = _boot(pa, np.random.default_rng(SEED), by_family=True)
        rows.append([name, len(pa), len({_fam(x) for x in pa.allele}),
                     num(_gap(pa), True), ci(*a), ci(*f)])
    out += ['**Resampling unit.** Intervals from resampling alleles, as reported, against '
            'intervals from resampling allele families, which does not treat related '
            'alleles as independent.', '',
            _tab(['Predictor', 'Alleles', 'Families', 'Gap', 'By allele', 'By family'], rows)]

    rows = []
    for name, col, _, mask in ARMS:
        sub = restrict(d, col, mask); cells = [name]
        for m in (5, 10, 15, 20):
            pa = per_allele(sub, col, 'roc_auc', min_cls=m)
            g = _gap(pa)
            nr = int(pa.is_rare.sum()) if len(pa) and 'is_rare' in pa.columns else 0
            cells.append(f'{len(pa)} / {nr} / '
                         + (num(g, True) if np.isfinite(g) else '-'))
        rows.append(cells)
    out += ['**Evaluability threshold.** Alleles admitted, of which rare, and the gap, as '
            'the minimum records of each class varies. The reported figures use ten.', '',
            _tab(['Predictor', 'min 5', 'min 10', 'min 15', 'min 20'], rows)]

    rows = []
    for name, col, _, mask in ARMS:
        pa = per_allele(restrict(d, col, mask), col, 'roc_auc')
        gm, ga = _gap(pa), _gap(pa, np.mean)
        rows.append([name, len(pa), num(gm, True), num(ga, True),
                     num(ga - gm, True) if np.isfinite(gm) and np.isfinite(ga) else '-'])
    out += ['**Summary statistic.** The gap as a difference of medians across alleles, as '
            'reported, against a difference of means.', '',
            _tab(['Predictor', 'Alleles', 'Median', 'Mean', 'Difference'], rows)]

    fold = pd.read_parquet(BASE / 'cv_folds.parquet')[KEY + ['fold']]
    df = d.merge(fold, on=KEY, how='left')
    rows = []
    for name, col, _, mask in ARMS:
        if 'retrained' not in name:
            continue
        sub = restrict(df, col, mask)
        pa_all = per_allele(sub, col, 'roc_auc')
        obs = [g for g in (_gap(per_allele(sub[sub.fold.eq(f)], col, 'roc_auc'))
                           for f in sorted(sub.fold.dropna().unique())) if np.isfinite(g)]
        if len(obs) < 3 or not len(pa_all) or pa_all.is_rare.sum() < 2:
            rows.append([name, num(_gap(pa_all), True), '-', '-', '-']); continue
        v, r = pa_all.value.to_numpy(), pa_all.is_rare.to_numpy()
        k = max(2, int(round(r.sum() / len(obs))))
        exp = [np.median(rng.choice(v[~r], min(20, int((~r).sum()))))
               - np.median(rng.choice(v[r], k)) for _ in range(NBOOT)]
        rows.append([name, num(_gap(pa_all), True), f'{np.std(obs):.4f}',
                     f'{np.std(exp):.4f}', f'{np.std(obs) / np.std(exp):.2f}'])
    out += ['**Cross-validation folds.** The reported figure pools out-of-fold predictions. '
            'The spread of the gap between folds is compared with the spread expected from '
            'resampling the pooled values to the rare alleles each fold retains; a ratio '
            'below one means the folds agree more closely than chance would produce.', '',
            _tab(['Predictor', 'Pooled gap', 'Observed s.d.', 'Expected s.d.', 'Ratio'], rows)]

    af = pd.read_csv(BASE / 'af_scores_final.tsv', sep='\t')
    af[['allele_norm', 'peptide']] = af.targetid.str.split('|', n=1, expand=True)
    m_ = af.merge(d[KEY + ['y', 'is_rare', 'n_records', 'editdist_alphafold']], on=KEY)
    m_ = m_[m_.editdist_alphafold >= 1]
    rows = []
    for c in [x for x in af.columns
              if af[x].dtype.kind == 'f' and af[x].notna().mean() > 0.5]:
        sub = m_.dropna(subset=[c])
        if len(sub) < 500 or sub[c].nunique() < 10:
            continue
        pa = per_allele(sub, c, 'roc_auc')
        if len(pa) >= 8:
            rows.append([c, len(pa), float(pa.value.median()), _gap(pa)])
    rows.sort(key=lambda r: -r[2])
    rows = [[c, n, num(m), num(g, True) if np.isfinite(g) else '-'] for c, n, m, g in rows]
    out += ['**Choice of confidence quantity.** Every quantity the fine-tuned model reports, '
            'scored under its own leakage exclusion. The reported arm uses the negated '
            'complex predicted aligned error.', '',
            _tab(['Quantity', 'Alleles', 'Median AUC', 'Gap'], rows)]

    return '\n'.join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description='Regenerate every reported evaluation figure.')
    ap.add_argument('--out', default='-', help='markdown output path, or - for stdout')
    ap.add_argument('--robustness', action='store_true',
                    help='also vary the resampling unit, evaluability threshold, '
                         'summary statistic, fold pooling and confidence quantity')
    ap.add_argument('--csv', default=str(BASE / 'results_per_allele.csv'),
                    help='tidy per-allele values, for auditing')
    args = ap.parse_args()

    d = load()
    s_full, l_full = collect(d, 'full')
    s_str, l_str = collect(d[d.af_cpae.notna()], 'structural')
    summary = pd.concat([s_full, s_str], ignore_index=True)
    long = pd.concat([x for x in (l_full, l_str) if len(x)], ignore_index=True)

    long.to_csv(args.csv, index=False)
    summary.to_csv(Path(args.csv).with_name('results_summary.csv'), index=False)

    parts = [
        '# Evaluation results', '',
        f'Generated by `evaluation_metrics.py`. Seed {SEED}, {NBOOT:,} bootstrap resamples '
        f'over alleles. Alleles require at least {MIN_CLS} records of each class for the rank '
        f'metrics and at least {MIN_AFF} exactly-measured records for the correlation. '
        'R-precision is the precision among the highest-scoring peptides of an allele, taking '
        'as many peptides as that allele has binders.', '',
        '## Leakage control', '', leakage_table(d),
        '## Full evaluation dataset', '',
        group_table(summary, 'full', 'roc_auc',
                    '**AUC-ROC.** Median per-allele, leakage-controlled.'),
        group_table(summary, 'full', 'pr_auc',
                    '**Precision-recall AUC.** Median per-allele. The baseline is the positive '
                    'rate, which differs between groups, so the gap column is not interpretable '
                    'and is given for completeness only.'),
        group_table(summary, 'full', 'r_precision',
                    '**R-precision.** Median per-allele. Precision among the highest-scoring '
                    'peptides, taking as many as the allele has binders.'),
        group_table(summary, 'full', 'spearman',
                    '**Spearman correlation with measured affinity.** Exact nanomolar records '
                    'only.'),
        depth_table(summary, 'full'),
        '## Structural subset', '',
        group_table(summary, 'structural', 'roc_auc',
                    '**AUC-ROC, structural subset.** Median per-allele, leakage-controlled.'),
        group_table(summary, 'structural', 'pr_auc',
                    '**Precision-recall AUC, structural subset.** Median per-allele.'),
        group_table(summary, 'structural', 'r_precision',
                    '**R-precision, structural subset.** Median per-allele.'),
        group_table(summary, 'structural', 'spearman',
                    '**Spearman correlation, structural subset.** Exact nanomolar records only.'),
        depth_table(summary, 'structural'),
        '## Structural mechanism', '', mechanism_table(), paired_structural(d),
    ]
    if args.robustness:
        parts += ['## Robustness', '', robustness_section(d)]

    text = '\n'.join(parts)
    if args.out == '-':
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f'written {args.out}', file=sys.stderr)
    print(f'per-allele values {args.csv}', file=sys.stderr)


if __name__ == '__main__':
    main()
