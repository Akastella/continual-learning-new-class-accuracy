# Cross-group current-image supervision

Run from the code-package root with Python 3.12 and NumPy 1.26.4. Each output directory must be new:

```powershell
python -B -I -X utf8 evidence/cross_group_supervision/recompute_endpoints.py --output cross_group_counts_reader
python -B -I -X utf8 evidence/cross_group_supervision/recompute_scores.py --output cross_group_scores_reader
```

The first command recomputes integer counts and every statistic in `statistics.json` from the 30 compact prediction files. Its `counts.csv` and `summary.json` are checked against the recorded results. Five paired seeds are the independent units. Six endpoints per seed include the three unchanged original comparisons and fifteen newly trained models. Means, sample SDs, individual seeds, paired effects, descriptive intervals, and the fixed H1/H2 decisions are retained. Models and image data are unnecessary for these calculations.

The second command uses the 30 small files in `score_analysis/inputs/`. Each contains only old/current winning scores and classes, labels, official row identities, class order, and the fixed calibration mask. It reconstructs the class-stratified PCG64 split: 2,000 calibration and 8,000 test images, with seed 2026110800 plus the original class. It calls the unchanged `calculate` and `measure` functions in `score_analysis/executed_analysis.py`, copying the small inputs into the new output directory. It recomputes all 180 offset attempts, unavailable targets, test violations, unadjusted scores, all per-seed/per-old-task rows, and `summary.json`; each is compared with its recorded counterpart. No model, full logits, original research directory, or protocol document is loaded.

`paired_effects.json` is computed from the newly generated offset CSV. It includes all pairs of the five intervention methods and the policy interaction, at every recorded target (5, 10, 15, 20, 25, 30 percent). Every statistic includes the actual common available seed IDs, n, values, mean, sample SD, and signs. All five seeds are available for the five methods at 5/10/15/20. `allq_seed_summary.json` also retains model summaries and all nine old tasks. The original three-target paired summary is retained as `score_analysis/recorded_paired_effects.json`; its values must be reproduced exactly. These descriptive comparisons do not establish equivalence, and actual test retention levels remain explicit in the CSVs.

The commands create their output directories and verification summaries. The package retains the original numerical reference files and unchanged inputs, including the reused baseline files.

`protocol.md` is a scientific-design excerpt of the original training plan, with its introductory administrative paragraph omitted. The unabridged original remains in the authors' working directory. The original execution records retain its input identifier; the excerpt is identified separately. `score_analysis/SCORE_PLAN.md` and `score_analysis/recorded_design.md` retain the complete recorded scalar-analysis design. The split is held out from scalar fitting; the complete CIFAR-100 test source was previously accessed in the broader project.

`execution/` supplies the original `run.py` and `reference_run.py` source files, input and execution records, environment, short train-only checks, all fifteen 70-epoch records (1,050 epochs), completed endpoint records and evaluation timing. `provenance.json` records identifiers for the original plan, source files, predictions, and locally retained states/logits. The unchanged training engine and portable entry are in `experiments/paa/cross_group_supervision/`. That directory's README gives training commands. The mixed-replay policy changes current/old exposure; it preserves the paired mixed-policy schedules. All-class accuracy uses old/new weights 0.9/0.1.
