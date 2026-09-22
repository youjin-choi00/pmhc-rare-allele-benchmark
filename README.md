# Applying AI to the Discovery of Cancer Antigens for Immunotherapy

Code accompanying an MSc dissertation, UCL


## Repository Structure

Each script resolves the project root without requiring path adjustments. This repository provides `scripts/`; `data/` sits alongside it:

    <root>/scripts/
    <root>/data/raw/          # IEDB MHC ligand export
    <root>/data/processed/    # Curated datasets, edit distances and scores

    
## Pipeline

    python scripts/build_dataset.py          # The curated and analysis datasets
    python scripts/edit_distance.py --arm mhcflurry
    python scripts/build_clustered_folds.py --max_dist 2
    python scripts/scoring/score_netmhcpan.py --version 4.1
    python scripts/evaluation_metrics.py --out logs/evaluation_metrics.md --robustness

`scripts/reproduce.sh` runs the full sequence in order.


## Scripts

| Script | Produces |
|---|---|
| `build_dataset.py` | The ten-step reduction from the IEDB export to 89,846 nine-mers across 116 alleles (Section 3.4), and the five-fold partition |
| `edit_distance.py` | Each record's Levenshtein distance to every predictor's training peptides (Section 4.1) |
| `build_clustered_folds.py` | The similarity-clustered partition the retrained arms are reported on (Section 4.2) |
| `scoring/` | One script per evaluation arm (Section 4.4) |
| `evaluation_metrics.py` | Every reported metric and confidence interval: 2,000 bootstrap resamples over alleles at seed 42 (Section 3.7) |

## Excluded

NetMHCpan 4.1 and 4.2 binaries (DTU academic license), model weights, and the raw IEDB MHC ligand export are not distributed within the repository.

