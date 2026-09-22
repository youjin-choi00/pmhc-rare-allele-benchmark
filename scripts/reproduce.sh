#!/usr/bin/env bash
# Rebuilds the scored outputs the dissertation reports, in order.
# The retrained arms are reported on similarity-clustered folds (Section 4.2);
# the identity-partitioned versions are retained for the comparison in that section.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_dataset.py
python scripts/build_clustered_folds.py --max_dist 2

python scripts/retraining/retrain_mhcflurry.py --all \
  --folds cv_folds_clustered.parquet --suffix _clustered
scripts/run_ms_cv_clustered.sh
python scripts/scoring/score_mhcseqnet2_cv.py \
  --experiment ms_cv_clustered --folds cv_folds_clustered.parquet --suffix _clustered

# the reported arms read the unsuffixed names
cp data/processed/scores_mhcflurry_cv_clustered.parquet  data/processed/scores_mhcflurry_cv.parquet
cp data/processed/scores_mhcseqnet2_cv_clustered.parquet data/processed/scores_mhcseqnet2_cv.parquet

python scripts/evaluation_metrics.py --out logs/evaluation_metrics.md --robustness

# Everything below reads the retrained arms, so all of it must be regenerated
# whenever the folds change. Omitting these is how the constructed-negative range
# and the ensemble results came to disagree with the scores they describe.
python scripts/pairwise_and_decoy.py  2>&1 | tee logs/pairwise_and_decoy.txt
python scripts/positional_decoys.py   2>&1 | tee logs/positional_decoys.txt
python scripts/ensemble.py            2>&1 | tee logs/ensemble.txt
python scripts/combination_strata.py  2>&1 | tee logs/combination_strata.txt
python scripts/locus_conditioned.py   2>&1 | tee logs/locus_conditioned.txt
