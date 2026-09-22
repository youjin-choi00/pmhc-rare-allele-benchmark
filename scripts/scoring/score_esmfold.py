#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE   = Path(__file__).resolve().parents[2]
OUT    = BASE / 'data/processed'
LINKER = 'G' * 25
OFFSET = 512
COLS = ['targetid', 'pep_plddt', 'groove_plddt', 'complex_plddt', 'ptm',
        'pae_complex', 'pae_interface']


def load_model():
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',
                                                 low_cpu_mem_usage=True)
    model.esm = model.esm.half()      # The language model in half precision, as in the
    model = model.cuda().eval()       # reference implementation; the trunk stays fp32
    model.trunk.set_chunk_size(64)
    return tok, model


def predict(tok, model, groove: str, peptide: str, pdb_path=None) -> dict:
    seq = groove + LINKER + peptide
    ng, nl, npep = len(groove), len(LINKER), len(peptide)
    enc = tok([seq], return_tensors='pt', add_special_tokens=False)
    pos = torch.arange(len(seq), dtype=torch.long)
    pos[ng + nl:] += OFFSET
    enc = {k: v.cuda() for k, v in enc.items()}
    enc['position_ids'] = pos.unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(**enc)

    if pdb_path is not None:
        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
        pdb_path.write_text(model.output_to_pdb(out)[0])
    plddt = out['plddt'][0, :, 1].detach().cpu().numpy()   # CA atom
    res = dict(pep_plddt=float(plddt[-npep:].mean()),
               groove_plddt=float(plddt[:ng].mean()),
               complex_plddt=float(np.concatenate([plddt[:ng], plddt[-npep:]]).mean()),
               ptm=float(out['ptm']) if 'ptm' in out else np.nan,
               pae_complex=np.nan, pae_interface=np.nan)

    pae = out.get('predicted_aligned_error')
    if pae is not None:
        m = pae[0].detach().cpu().numpy()
        keep = np.r_[0:ng, ng + nl:ng + nl + npep]          # drop the linker
        sub = m[np.ix_(keep, keep)]
        res['pae_complex'] = float(sub.mean())
        res['pae_interface'] = float(np.concatenate([
            sub[:ng, ng:].ravel(), sub[ng:, :ng].ravel()]).mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True, help='targets tsv in data/processed')
    ap.add_argument('--out', required=True, help='output tsv in data/processed')
    ap.add_argument('--limit', type=int, default=0, help='stop after N, for timing')
    ap.add_argument('--pdb_dir', default='', help='if set, write predicted structures here')
    args = ap.parse_args()

    tgt = pd.read_csv(OUT / args.targets, sep='\t')
    tgt[['groove', 'peptide']] = tgt.target_chainseq.str.split('/', n=1, expand=True)
    outp = OUT / args.out

    done = set()
    if outp.exists():
        done = set(pd.read_csv(outp, sep='\t').targetid)
        print(f'resuming: {len(done):,} already done')
    todo = tgt[~tgt.targetid.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f'{len(todo):,} to predict of {len(tgt):,} targets', flush=True)
    if not len(todo):
        return

    tok, model = load_model()
    if not outp.exists():
        outp.write_text('\t'.join(COLS) + '\n')

    t0 = time.time()
    with open(outp, 'a') as fh:
        for i, r in enumerate(todo.itertuples(), 1):
            pp = (Path(args.pdb_dir) / (r.targetid.replace('|','_').replace('*','')
                  .replace(':','') + '.pdb')) if args.pdb_dir else None
            res = predict(tok, model, r.groove, r.peptide, pp)
            fh.write('\t'.join([r.targetid] + [f'{res[c]:.6f}' for c in COLS[1:]]) + '\n')
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                rate = el / i
                left = rate * (len(todo) - i)
                print(f'  {i:,}/{len(todo):,}  {rate:.1f} s/structure  '
                      f'{left/3600:.1f} h remaining', flush=True)

    el = time.time() - t0
    print(f'\n{len(todo):,} structures in {el/60:.1f} min ({el/len(todo):.1f} s each)')
    print(f'full set of {len(tgt):,} would take {el/len(todo)*len(tgt)/3600:.1f} h')
    print(f'saved {outp}')


if __name__ == '__main__':
    main()
