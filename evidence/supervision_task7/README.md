# Task-7 current-image supervision data

This package contains five paired seeds and six endpoints at the CIFAR-100 task-7 boundary, with 70 old and 10 current classes. The four added conditions are old-only current10 restriction, an old-only 0.05 all80 current component, mixed-replay full incoming loss, and mixed-replay restricted incoming loss. Original full-loss ER and current-only endpoints are reused from the same pre-task states.

Run the numerical recomputation with Python and NumPy from this directory.

```text
python -B -s recompute.py --output recomputed_task7
```

The script verifies the supplied summary and writes regenerated CSV and JSON statistics to a new output directory. It uses the 30 NPZ files in `reader_data/`. Each NPZ retains all 10,000 official test row identities, original and mapped labels, the seed class permutation, all80/current10 predictions, and old/current group scores and predictions. Raw endpoint metrics retain the 8,000 images in mapped classes 0–79. `counts.csv` contains integer counts, and `seed_scores.csv` includes every raw endpoint score.

Calibration uses the first 20 of each original class's 100 official indices after a PCG64 permutation with seed 2026110800 + original class. Test uses the remaining 80. Restricting each seed to its seen 80 classes gives 1,600 calibration and 6,400 test images. Old calibration has 1,400 images and old test has 5,600. Seen-class image identities vary across seeds. All conditions within one seed use identical image identities. The official image source has already been used in the broader project.

Each model receives one scalar added to all current-class logits. Calibration selects the largest scalar satisfying old accuracy at least q+3 pp, with old/current ties favoring old. Targets remain 5, 10, 15, 20, 25 and 30 percent. The q20 requirement is 322 old-correct calibration predictions. `score_counts.csv` retains all target attempts, availability, violations, calibration/test counts, and seven old-task scores. `unadjusted_scores.csv` retains calibration/test scores before adjustment. `summary.json` includes paired means, sample SDs, each seed effect, fixed H1/H2 decisions, q20 comparisons, and target availability. Every boundary uses five seed units. Conditional score summaries state how many of those five are available.

Raw H1 uses the mixed-minus-old-only change in restriction effects, requiring positive new-accuracy and negative competition-loss interactions in paired mean and at least three seeds. H2 fixes the coefficient at 0.05 and requires new accuracy at least half original ER and old accuracy above original ER in paired mean and at least three seeds. Integer count arithmetic precedes division. Mixed replay retains 175,000 old presentations and adds 175,000 current revisits per model, while old-only replay retains 350,000 old presentations.

## Train one endpoint

`training/train.py` accepts a supplied original `pre_task_07.pt`, the official CIFAR-100 data directory, and a fresh output directory. Select one condition and seed explicitly.

```text
python -B -s -u -X utf8 training/train.py PRE_TASK_07.pt DATA_ROOT NEW_OUTPUT --seed 3101 --condition old_restricted
```

The other conditions are `old_blend05`, `mixed_full`, and `mixed_restricted`. Use `--check-inputs-only` to validate checkpoint content on CPU. Use CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the versions in `../../experiments/paa/cross_group_supervision/source/requirements-r9-windows-lock.txt`. On Windows, point TEMP and TMP to a writable directory with sufficient space. DATA_ROOT contains the official archive plus extracted train/meta files. Downloads are disabled.

The entry restores the supplied pre-state directly and executes the fixed 70 epochs, preserving task-7 sampling, augmentation, fresh optimizer, learning-rate schedule and loss scale. It saves one final model and epoch records. `task7_math.py` contains exact copies of the executed task-7 loss and training wrapper, and `../../experiments/paa/cross_group_supervision/` supplies the shared sampling and training code. All decisions use the first 80 outputs of the original 100-row head. Training accepts explicit file paths and runs independently of the internal execution control files. The package supplies prediction data and code, while the original checkpoints and images are provided separately.
