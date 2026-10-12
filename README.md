# Routing-conditional calibration reproduction

This repository reproduces and stress-tests *Can We Trust Jev After Routing?*.
It prepares Civil Comments and GoEmotions, obtains typed decision
probabilities, performs the score-conditioned audit with paired bootstrap
uncertainty, applies BH-FDR, runs random/oracle controls, and generates all
paper tables and figures. The robustness pipeline adds absolute calibration,
bin-count sensitivity, continuous restricted-cubic-spline calibration curves,
within-bin permutation tests, soft-label analysis, prompt sensitivity, and a
second-model comparison.

## New: score-resolution and held-out follow-up audit

The CPU-only `strengthen` command adds local score balancing, fitted-null
binning diagnostics, held-out recalibration with full refitting bootstrap,
and all 12 directed label pairs. It reuses existing OpenJev/OneJev predictions,
does not require another model run, and writes separate resumable results.
It does **not** reuse the legacy continuous-spline uncertainty estimates.

Full offline run (2,000 resamples; all configured prediction sets):

```bash
bash scripts/run_strengthening_full.sh configs/reproduce.yaml results/strengthening_v1 8
```

See [the full command guide and statistical protocol](docs/STRENGTHENING.md)
for local prediction paths, assumptions, outputs, report regeneration, and
the archive to send back for review. The only new config is
`configs/strengthen.yaml`; keep your existing server-specific paths in
`configs/reproduce.yaml`.

## Decisions made for underspecified details

All decisions live in `configs/reproduce.yaml`; its SHA-256 is written into
every run manifest.

- Civil Comments soft targets are binarized at `>= 0.5`.
- Routing selects `ceil(r * N)` rows. Scores are descending; `example_id` is
  the deterministic tie-break.
- Score bins are pooled downstream-score quantiles. Routed and non-routed
  examples use the same boundaries. Identical rounded scores are never split,
  so fewer than ten effective bins are possible.
- A bin is supported only with at least 50 routed and 50 non-routed examples.
- Conditional bin differences are weighted by routed count.
- The raw comparison is routed minus the full sample, following Appendix A.1.
- Each paired bootstrap resample re-ranks routing scores and reconstructs bins.
- The two-sided bootstrap sign p-value is
  `2 * (min(n_boot <= 0, n_boot >= 0) + 1) / (B + 1)`, capped at one. With
  2,000 samples its minimum is approximately 0.001, consistent with the
  draft's repeated `q=0.0011` after BH over 16 tests.
- Unsupported planned GoEmotions cells have no point estimate, CI, or raw
  p-value and remain in the 12-test BH family with `p=1`.
- OpenJev uses the model card's online-FP8 vLLM recipe and calibration
  constants.

### Important data discrepancy

The pinned public `google/civil_comments` test parquet has 97,320 rows. It has
348 duplicate occurrences when all eight published fields are compared, and
458 duplicate texts. Neither matches the draft's statement that 88 rows were
removed. The default uses exact-row deduplication because it is deterministic
and does not selectively inspect target labels. The preparation manifest will
therefore report 96,972 rows. If the original 97,232-row source table becomes
available, point `paths.civil_input` at it and set deduplication to `none`, or
select an explicit `id` rule. The pipeline never silently deletes an arbitrary
88 rows just to match the stated count.

## Internet-connected preparation

Run this on Linux x86_64 with Python 3.12, matching the offline server:

```bash
chmod +x scripts/*.sh
scripts/download_online.sh /transfer/offline_bundle
```

The bundle includes pinned model/data snapshots, Python wheels, and checksums.
The model snapshot is about 55 GB. Verify a clean offline installation before
moving it to the server.

## Offline installation

```bash
scripts/install_offline.sh /transfer/offline_bundle .venv
source .venv/bin/activate

mkdir -p data/raw
ln -s /transfer/offline_bundle/data/civil_comments data/raw/civil_comments
ln -s /transfer/offline_bundle/data/go_emotions data/raw/go_emotions
```

The host needs a sufficiently recent NVIDIA driver for B200/CUDA 13.0. The
Python wheels provide their CUDA runtime; a full host CUDA toolkit is not
required for the prebuilt path.

## Run

Prepare datasets:

```bash
jevcal --config configs/reproduce.yaml prepare
```

In terminal 1, start OpenJev on one B200:

```bash
scripts/serve_one_b200.sh /transfer/offline_bundle/models/openjev
```

The default config uses the single helper endpoint `http://127.0.0.1:3000`.
`serve_two_b200.sh` remains available for an optional two-replica deployment;
add port `3001` back to `model.endpoints` when using it.

In terminal 2, run resumable inference and analysis:

```bash
jevcal --config configs/reproduce.yaml infer --dataset all
jevcal --config configs/reproduce.yaml analyze
```

Inference checkpoints each completed example in JSONL. Restarting the same
command resumes without repeating successful rows. For a short end-to-end
analysis smoke test, use `analyze --quick`; this does not produce paper-grade
confidence intervals.

Outputs are placed in:

- `data/prepared/`: normalized examples and provenance manifests;
- `artifacts/predictions/`: labels plus cached OpenJev probabilities;
- `results/tables/`: CSV and Parquet tables;
- `results/bootstrap/`: compressed bootstrap draws;
- `results/figures/`: the two main plots;
- `results/manifest.json`: hashes and the complete analysis protocol.

## Robustness analyses

The main prediction files are reused for every robustness check that does not
change the model prompt. No GPU is needed for absolute calibration, binning,
spline, permutation, or soft-label analyses.

### 1. Generate OpenJev prompt variants

Keep the OpenJev server running, then run the two additional frozen prompt
sets. Each command has its own resumable checkpoint and never overwrites the
primary predictions.

```bash
jevcal --config configs/reproduce.yaml infer \
  --dataset civil_comments \
  --run-name openjev_minimal \
  --prompt-set minimal

jevcal --config configs/reproduce.yaml infer \
  --dataset civil_comments \
  --run-name openjev_dataset_aligned \
  --prompt-set dataset_aligned
```

### 2. Generate OneJev predictions

Download the model and its serving package on an internet-connected machine.
Both revisions are required so the final run cannot silently track a moving
branch:

```bash
scripts/download_onejev_online.sh \
  /transfer/offline_bundle \
  aec969d68e7e8e8b0df051e065c04304c0cd7163 \
  81ce62f1597c91e46767d4d02d6ac2e18534fe94
```

On the offline server, install the generated `qev` wheel if it is not already
installed. Stop OpenJev first because OneJev's default endpoint uses port 8000.

```bash
python -m pip install --no-index \
  --find-links /transfer/offline_bundle/wheels 'qev[torch]'

GPU_ID=0 scripts/serve_onejev_b200.sh \
  /transfer/offline_bundle/models/onejev-27b
```

The pinned OneJev model revision is already recorded under
`model.profiles.onejev.revision` in `configs/reproduce.yaml`. In another terminal:

```bash
jevcal --config configs/reproduce.yaml infer \
  --dataset civil_comments \
  --run-name onejev_criteria_rich \
  --model-profile onejev
```

### 3. Run every statistical robustness check

For a short smoke test:

```bash
jevcal --config configs/reproduce.yaml robustness --quick
```

For final paper results, require every configured prompt/model prediction set:

```bash
jevcal --config configs/reproduce.yaml robustness --strict-prediction-sets
```

This produces:

- `results/robustness/tables/absolute_calibration.*`: prevalence, mean score,
  signed error, Brier, ECE, ICI, calibration intercept, and slope for each
  routed/non-routed group;
- `binning_sensitivity.*`: 5/10/20-bin estimates with paired-bootstrap CIs;
- `continuous_spline.*` and `continuous_spline_curves.*`: continuous
  group-specific calibration curves, integrated shifts, and interaction tests;
- `within_bin_permutation.*`: matched-score permutation-null p-values;
- `label_sensitivity.*`: hard labels versus Civil Comments soft annotations;
- `prediction_set_comparison.*`: prompt/model estimates and CIs;
- `results/robustness/figures/figure3_*.png` through `figure6_*.png`;
- `results/publication_tables/*.tex`: ready-to-include LaTeX tables.

The soft-label Brier calculation is the expected Bernoulli score
`p^2 - 2*p*y + y`; it is not the squared distance `(p-y)^2`.

### 4. Regenerate tables and figures only

After editing plot styles or copying final CSV files, rebuild every figure and
LaTeX table without inference or resampling:

```bash
jevcal --config configs/reproduce.yaml report
```

### Full final sequence

Assuming prepared data and all four prediction sets already exist:

```bash
jevcal --config configs/reproduce.yaml analyze
jevcal --config configs/reproduce.yaml robustness --strict-prediction-sets
jevcal --config configs/reproduce.yaml report
```

## Tests

```bash
pytest
```

The tests cover label preparation, deterministic routing, tied-score binning,
conditional weighting, bootstrap p-values, and BH correction without requiring
a GPU or model download.
