# Data and code availability

## Observational data

This study uses measured surface-radiation and meteorological observations from the Baseline Surface Radiation Network (BSRN). The source release is:

Lanconelli, C. et al. (2023). *Baseline surface radiation data snapshot 2023-03-31*. PANGAEA. https://doi.org/10.1594/PANGAEA.957398

The release contains quality-controlled station files and their source quality flags. Its overall temporal coverage is 1992 to 2023. The benchmark retained 51 stations having at least 60 consecutive monthly files and used fixed station-disjoint groups of 33 training, 7 validation, and 11 test stations.

The source data are governed by the Baseline Surface Radiation Network License 1.0. The BSRN release conditions require users to acknowledge the archive and relevant stations and restrict passing the datasets to third parties without WRMC agreement. For this reason, neither raw BSRN archives nor processed Parquet files are included in this repository.

Users should obtain the observations directly from PANGAEA. The repository provides the downloader, archive checksums, fixed station manifest, preprocessing code, and audit summaries required to reconstruct the analysis data. The BSRN conditions are available at https://bsrn.awi.de/data/conditions-of-data-release/.

## Reproducibility metadata

The following files can be shared because they contain protocol, provenance, or integrity information rather than measured time-series observations:

- `manifests/final_station_manifest.csv`: station membership and fixed train, validation, and test assignments
- `manifests/station_coverage.csv`: month-level availability summary used for cohort selection
- `manifests/checksums.sha256`: SHA-256 hashes of the 76 source archives
- `manifests/audit_summary.json`: source audit summary
- `manifests/preprocessing_report.json`: preprocessing counts and settings
- `manifests/processed_integrity_audit.json`: integrity checks for the reconstructed 15-minute files
- `freezes/`: the pre-test freeze manifest and its hash

## Repository material

The repository at https://github.com/vsidaarth/physics-mamba-zero-shot-forecasting contains:

- the complete data download, QC, aggregation, training, evaluation, and analysis code used for the study;
- the frozen experimental protocol and sealed-test audit;
- trained model checkpoints and their metadata;
- station-level and aggregate result tables needed to verify the reported values; and
- the publication figures and scripts that regenerate them.

No target-station observations were used for retraining or adaptation. Test observations are required only to independently evaluate the frozen models.
