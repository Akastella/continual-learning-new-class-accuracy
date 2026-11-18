# Fixed-exposure replay-current supervision control

Run `python -B -s recompute.py --output recomputed_repeats` from this directory with Python and NumPy. The script recomputes integer counts, raw and 20% target paired statistics, and the stated half-effect criteria from the 40 NPZ files in `reader_data/`.

Each task boundary has five paired seed units. Task 7 predicts over 80 seen classes and task 9 over 100. A is old-only restricted, B is mixed-restricted, and C is the added revisit_current10 condition. Relative to B, C changes only the repeated-current replay loss to current10 CE. Scheduled current and old replay losses, images, order, augmentation, exposure, whole-batch mean, 70 epochs and optimizer events are retained. Original full-loss ER supplies an additional descriptive reference.

The fixed new-accuracy surplus is (A_N + B_N - 2 C_N)/2. The fixed competition-loss surplus is (2 C_L - A_L - B_L)/2, where L is current10 minus all-seen new accuracy. Each requirement passes with a nonnegative paired mean and at least three nonnegative seeds. Equality meets the threshold. Both requirements are necessary. Arithmetic adds integer correct counts before scaling. All seeds, including failures, remain in summary.json and the CSVs.

Raw metrics retain only each seed's seen classes. q20 uses the original-class PCG64 split with seed 2026110800 + original class, selecting 20 indices for calibration and 80 for test. Task 7 has 1600/6400 calibration/test images and 1400 old calibration images, requiring 322 old-correct calibration predictions. Task 9 has 2000/8000 images and 1800 old calibration images, requiring 414. The fixed margin is 3 pp. Current-logit offsets are fitted to calibration only. Old/current ties favor old. Availability and test old-target violations are retained. q20 and within-group comparisons are descriptive, with no prespecified adjusted ordering. Image identities match across conditions within a seed, while seen-class sets can differ across seeds. The official image source was previously used in the project.

`counts.csv` contains integer raw counts. `seed_scores.csv` and `unadjusted_scores.csv` contain every seed endpoint and calibration/test score. `q20_scores.csv` contains fitted offsets, counts, availability, violations and old-task scores. `split_identities.json` preserves the official calibration/test row indices.

To train one added endpoint, supply the original ER pre-task checkpoint, the CIFAR-100 data directory, and a new output directory. Use the CUDA environment described in the repository README.

```text
python -B -s -u train.py PRE_TASK_07.pt DATA_ROOT NEW_OUTPUT --task-index 7 --seed 3101
```

For the tenth task, supply the corresponding checkpoint and use `--task-index 9`. Add `--check-inputs-only` to read the supplied pre-task state on CPU. Original states and data are generated using the repository's CIFAR-100 commands.
