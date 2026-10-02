# Doubling-time baselines

This folder evaluates three baselines against the fixed assembly-level splits in
`dse/data/ribosomal/`. No model is fit using validation or test labels.

| Method | Inputs | Fitted on this project's train split? |
| --- | --- | --- |
| `train_mean` | No features; predicts the mean training `log_dob_h` | Yes |
| `temperature_linear` | `growth_tmp` only; ordinary least squares with an intercept | Yes |
| `grodon` | Annotated genome-wide coding sequences and ribosomal-protein labels | No; uses gRodon's published model |

The primary score is mean squared error (MSE) of `log_dob_h` after applying the
training-set standardization in `normalization_stats_std_norm.json`. This is the
same target scale and loss function used by the sequence regression trainer. The
CSV also reports RMSE and MAE in standardized and unstandardized log-hour units.
Lower is better. Each assembly is counted once.

## 1. Run the two baselines available from the repository

From the repository root:

```bash
python experiments/baselines/evaluate_baselines.py
```

This needs only Python's standard library. It writes:

- `experiments/baselines/results/metrics.csv`: aggregate scores by split.
- `experiments/baselines/results/predictions.csv`: one prediction per assembly and method.
- `experiments/baselines/results/summary.json`: split sizes, normalization and fitted coefficients.
- `experiments/baselines/results/REPORT.md`: a short table and interpretation for team review.

The evaluator preserves an existing `REPORT.md` so team edits are not lost.
Pass `--overwrite-report` only when you intend to replace it with generated text.

The evaluator checks required columns, finite values, unique assembly IDs,
disjoint splits, and agreement between raw and standardized CSVs. The checked-in
data currently contains 301 train, 25 validation, and 74 test assemblies.

## 2. Obtain gRodon's inputs

The sequence column in the project CSV is **not** an annotated set of coding
genes. gRodon needs a CDS FASTA for each assembly, with headers identifying
ribosomal proteins. Install the [NCBI Datasets CLI][ncbi-cli], then pilot three
assemblies:

```bash
python experiments/baselines/download_cds.py --limit 3
```

After reviewing `experiments/baselines/cds/download_manifest.csv`, download the
rest with:

```bash
python experiments/baselines/download_cds.py
```

The downloader calls `datasets download genome accession ... --include cds`,
extracts the annotated CDS FASTA to `cds/<assembly_id>.fna`, and reuses files
already downloaded. The manifest records missing CDS or ribosomal annotations.
The `cds/` directory is ignored by Git because the downloads may be large.
Use `--metadata-only` to check all assembly annotation statuses without
downloading new FASTA files.
The first audit is summarized in `results/FEASIBILITY.md`.

## 3. Run gRodon

Install R with the `gRodon` and `Biostrings` packages as described in the
[gRodon2 repository][grodon]. Then run:

```bash
Rscript experiments/baselines/run_grodon.R
```

On the current Windows machine, R 4.6.1 and gRodon 2.7.3 are already installed
under the ignored `experiments/baselines/bin/` directory on D:. From the repo
root in PowerShell, use:

```powershell
& .\experiments\baselines\bin\portable-r-4.6.1-win-x64\bin\Rscript.exe .\experiments\baselines\run_grodon.R
```

The exact R and package versions used for the results are in
`results/R_ENVIRONMENT.md`. The local R package installation is machine
specific and ignored by Git; collaborators can install R and packages using
the upstream instructions.

For a short pilot, pass `--limit 3 --output experiments/baselines/results/grodon_pilot.csv`.

This writes `experiments/baselines/results/grodon_predictions.csv`, one row per
attempted assembly. A failed assembly has a status and error message rather than
a fabricated prediction. Re-running resumes completed rows; pass
`--retry-failed true` to retry failures after fixing their inputs.

Defaults are `mode=full`, `training_set=madin`, and no temperature correction.
gRodon's temperature option calls for *optimal growth temperature*. Use
`--temperature-source growth_tmp` only if the team confirms that the CSV's
`growth_tmp` has that meaning. For incomplete genomes, consider a separately
named run with `--mode partial` after checking assembly quality. Do not mix
settings in one output CSV.

gRodon returns predicted minimal doubling time (`d`) in hours. The evaluator
uses `log(d)` and then applies the project's **training** mean and standard
deviation for `log_dob_h`. The checked-in raw values appear to use natural log
(for example, `1.791759469` corresponds to 6 hours), but the original target
construction should be confirmed with the data owner before publication.

## 4. Score gRodon alongside the other baselines

```bash
python experiments/baselines/evaluate_baselines.py --grodon-predictions experiments/baselines/results/grodon_predictions.csv
```

The `all` cohort scores the mean and temperature methods on every row. The
`grodon_matched` cohort scores **all three methods on exactly the assemblies
with valid gRodon predictions**. Use this cohort for a three-way comparison;
report gRodon coverage from `summary.json`. Missing gRodon results must not be
silently dropped from only one comparator.

The completed run has 382 valid gRodon predictions: 285/301 train, 25/25
validation, and 72/74 test. The other 18 assemblies lack annotated CDS in
NCBI Datasets. See `results/REPORT.md` for scores and `results/FEASIBILITY.md`
for the input audit.

## Notes for the paper

- The sequence model takes the repository's sequence and temperature as input.
  gRodon uses genome-wide annotated coding genes, so the methods have different
  input information. Describe this difference in the comparison.
- gRodon estimates maximal growth potential, or minimal doubling time. Confirm
  that the project's `log_dob_h` labels represent a comparable biological target.
  The gRodon documentation also warns that estimates for slow growers can
  saturate above roughly five hours.
- The model's Frontier evaluation uses `DistributedSampler` with 32 data
  parallel ranks. Validation has 25 rows and test has 74, so sampler padding
  can duplicate rows. For the final model comparison, score predictions for
  each unique assembly once using the validation-selected checkpoint.
- Save the exact NCBI Datasets CLI and gRodon package versions used. Check for
  overlap between the project organisms and gRodon's external training data
  when interpreting its score.
- Cite the original [gRodon paper][grodon-paper] for the prokaryotic baseline.

[ncbi-cli]: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/download/genome/
[grodon]: https://github.com/jlw-ecoevo/gRodon2
[grodon-paper]: https://doi.org/10.1073/pnas.2016810118
