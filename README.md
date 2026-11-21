# Code and statistical materials

This package accompanies the article *How Replay Supervision Changes New-Class Accuracy and Retention*. It contains the original experiment code, configurations, and saved numerical evidence. The reported statistics can be recomputed on a CPU without downloading images or installing PyTorch. Training the networks again is a separate, substantially longer GPU task.

## Results in the paper

| Comparison | Inputs and commands |
| --- | --- |
| Fixed exposure in CIFAR-100, including the transfer between images on the same update | [Remove the group loss](evidence/revisit_supervision/) · [Move the loss](evidence/shift_current_loss/) |
| CIFAR-10 timing, including the common-start omission and weight reversal | [CIFAR-10 timing](evidence/reservoir_supervision_timing/) |
| Timing in Fashion-MNIST and twenty-task CIFAR-100 | [Fashion-MNIST](evidence/fashion_placement/) · [CIFAR-100](evidence/er_ace_timing_c100/) |
| Offset transfer at matched training exposure | [ciFAIR-100 replacement images](evidence/cifair_replacements/) |

## Recompute the retention-target score comparison

Three further sets of trained comparisons are supplied as plain NumPy-readable files:

```powershell
python -B -I -X utf8 evidence/classifier_initialization/recompute.py
python -B -I -X utf8 evidence/asymmetric_loss/recompute.py
python -B -I -X utf8 evidence/asymmetric_transfer_c100/recompute.py
```

The classifier-initialization comparison includes all 80 new models, corresponding original current-task counts, and same-model old-class evaluation. The asymmetric-loss comparison includes the 200 usual/current-only endpoints of four methods across five seeds and five tasks, with all earlier-task predictions. Each directory documents its arrays, available original parameter states, and seed aggregation. The CIFAR-100 intervention supplies 15 final-task endpoints on five paired seeds. It changes only the current-image candidate classes during the final 70-epoch update from each existing ER pre-task state, retaining the original replay and loss weighting. Its reported task-9 results are separate from the original tasks-6–9 averages. Full network parameters are unnecessary for these numerical calculations.

The added journal comparison uses the same validation images and existing networks. Its complete numerical inputs include all 300 validation candidates and 600 test outputs across CIFAR-10.1 and CIFAR-10.2. From this directory, run:

```powershell
python -B -X utf8 tools/recompute_score_frontier.py --artifact . --data evidence/score_frontier --output recomputed_score_reader
```

This NumPy-only command recomputes scalar choices, all selected old/current/all accuracies, test violations, and the old-class accuracy bounds. It does not require the original research directory, images, checkpoints, or PyTorch. The chosen scalar values were obtained on validation images; test images are only scored. Complete test curves describe the stored models and are not used to choose scalar values.

For the 35% validation requirement and 30% test target, run
`python -B -s tools/compare_fixed_margin.py --artifact . --data evidence/score_frontier --output recomputed_margin`.
The program reports eligibility, scores, and misses for every policy.
Its paired summary compares policies on the same eligible seeds.

## Recompute the other reported statistics

Two further numerical summaries describe the replay image counts and the individual earlier tasks of the selected models:

```powershell
python -B -X utf8 tools/count_replay_presentations.py --output replay_presentations_reader.json
python -B -X utf8 tools/summarize_selected_retention.py --data evidence/score_frontier --output selected_retention_reader
```

The training comparison matches scheduled current-stream minibatches and optimizer updates. Replay also reuses some images from the current task, so its total current-image exposure and computation are larger. The retention target in the model-selection comparison pools old images. The individual-task CSV makes the corresponding distribution across earlier tasks explicit.

The complete paired four-/sixteen-epoch comparison also uses the recorded intermediate evaluations, without repeating training:

```powershell
python -B -X utf8 tools/compare_four_budgets.py --history16 evidence/history_controls/full_history16 --matched-three evidence/four_budgets/matched_three.json --output four_budgets_reader
```

The first four epochs of a sixteen-epoch current-task run use the same fixed learning rate, optimizer, and scheduled examples as stopping after four epochs from those task-start parameters. Earlier tasks still have sixteen epochs. The resulting table separates preceding-task training from current-task training; all reported differences first average tasks within each paired seed.

Run these commands from this directory with Python 3.12. The only dependency for these calculations is NumPy 1.26.4.

```powershell
python -m pip install -r requirements-review.txt
python -B -I -X utf8 tools/recompute_all.py --output recomputed_reader
```

The output directory must not already exist. The command creates the numerical summaries and separate analysis logs. For an isolated environment, create a virtual environment first and use its Python executable for both commands.

The command runs eleven CPU analyses and writes a separate `command_N.log` for each:

| Analysis | Main output, relative to the chosen output directory |
| --- | --- |
| Primary paired statistics, uncertainty intervals, and external-test selection | `primary_cifar102/report.json` |
| Main accuracy, retention, training-budget, DER++, and Continual Backpropagation comparisons | `ijcnn/` |
| Full-training CIFAR-100 comparison | `cifar100/analysis.json` |
| Loss-weight and memory-content comparisons | `weight_and_memory.json` |
| ResNet-32 comparison | `resnet32.json` |
| Common selection across thirty candidates | `weight_retention.json` |
| Sixteen-epoch training throughout the history | `full_history16/` |
| Training with only the classes encountered so far in the loss | `seen_classes4/` |
| Validation fitting of an additive score correction | `validation_calibration/`, fit files |
| Test evaluation of that fitted correction | `validation_calibration/`, test files |
| History-budget comparisons on the five shared seeds | `matched_history_budgets.json` |

Within `ijcnn/`, `accuracy_summary.csv` contains aggregate accuracies; `paired_comparisons.csv` and `paired_seed_values.csv` contain paired summaries and individual seed values; `endpoint_accuracies.csv` contains individual model scores; and `selection.csv` contains model-selection results. Each seed is averaged over the specified tasks before computing paired differences and sample standard deviations. The two history controls each contain 120 evaluated models and 480 rows of integer counts.

The package includes integer counts and saved per-example predictions, labels, official sample indices, and task identifiers where used by these analyses. Compact prediction files retain both the original argmax and the prediction among current classes. Some external-test analyses also include saved logits. These commands recompute statistics from saved evidence; they do not rerun network inference or training. The research workspace retains selected model states for the main comparisons and the original numerical outputs. The training commands recreate other model states.

## Recompute the cross-group supervision comparison

The added CIFAR-100 supervision experiment trained fifteen 70-epoch final-task models (1,050 epochs) on the same five paired pre-task states. It compares incoming-loss restriction under old-only and mixed replay, and a fixed 5% all-class component on current images. Its saved numerical evidence is recomputed separately from the original eleven-command run:

```powershell
python -B -I -X utf8 evidence/cross_group_supervision/recompute_endpoints.py --output cross_group_counts_reader
python -B -I -X utf8 evidence/cross_group_supervision/recompute_scores.py --output cross_group_scores_reader
```

Both commands need only NumPy and new output directories. The first reproduces all 30 integer-count rows and the complete seed-level and paired statistics. The second reads 30 compact group-score NPZs, verifies the fixed 2,000/8,000 calibration/test identities, and reproduces all 180 offset attempts (100 available), complete per-seed/per-old-task results, unavailable targets, violations, and summaries. It also generates every pair of the five methods and the policy interaction at 5/10/15/20 percent with n=5, plus all remaining recorded targets with their actual available seeds. Means and sample SDs are computed across paired seeds. The original paired-effect values are checked exactly; these descriptive comparisons do not imply equivalence. The commands create their output directories; the original numerical reference files are included in `evidence/cross_group_supervision/`. That directory's README describes the scientific protocol excerpt, complete score-analysis design, original source files, and 1,050 epoch records. Full models, full logits, images, and PyTorch are unnecessary for these calculations.

## Recompute the earlier CIFAR-100 boundary

The same two fixed supervision predictions were tested at the eighth task of the five histories. The four added conditions trained 20 models for 70 epochs each. Original full-loss ER and current-only models were reused. Training and prediction use the 80 seen classes, with 70 old and ten current classes. Each boundary has five paired seed units.

```powershell
python -B -s evidence/supervision_task7/recompute.py --output recomputed_task7
```

The package supplies 30 compact NPZs, integer counts, all fixed target choices, old-task scores, and paired statistics. It uses 1,600 calibration and 6,400 test images under the same per-original-class split rule and three-point validation margin. Image identities match across conditions within each seed and vary with the seed's seen-class set. The official image source was already used in the study. See `evidence/supervision_task7/README.md` for the single-condition training command. Generate its task-start state with the original CIFAR-100 history commands below or supply that saved state explicitly.

## Repeated images, replacement images, and validation objectives

```text
python -B -s evidence/revisit_supervision/recompute.py --output recomputed_repeats
python -B -s evidence/cifair_replacements/recompute.py --output recomputed_cifair
python -B -s tools/compare_validation_objectives.py --artifact . --output recomputed_objectives
```

The first command reproduces the fixed-exposure result for ten added models and the original comparisons at both CIFAR-100 task boundaries. Its [training instructions](evidence/revisit_supervision/README.md) use the same replay sampler and training engine as the mixed-replay experiment. The second recomputes the evaluation on ciFAIR-100 replacements using the existing CIFAR-100 offsets. Its [data and inference instructions](evidence/cifair_replacements/README.md) give the original dataset link and the CPU prediction command. The third compares new-class and overall validation accuracy using the same 30 models, validation labels, and feasible offsets. It reproduces the previous choices for the new-class objective at both validation thresholds. All three commands require only Python and NumPy for recomputation.

The placement comparison moves the all-class loss from repeated to nonrepeated current images while retaining the mixed training stream. The CIFAR-10 comparison uses the original reservoir and later retrieval. Recompute both from the saved group scores:

```text
python -B -s evidence/shift_current_loss/recompute.py --artifact . --output relocated.json
python -B -s evidence/reservoir_replay_loss/recompute.py --output reservoir.json
python -B -s tools/compare_old_accuracy.py --artifact . --output original_and_replacement.json
```

The [placement instructions](evidence/shift_current_loss/README.md) reuse the same training entry as the fixed-exposure experiment. The [reservoir instructions](evidence/reservoir_replay_loss/README.md) use the original CIFAR-10 history and memory files. The last command compares old-accuracy bounds and the same original and replacement test positions, and reports group AUROC and test-sample accuracy at measured 20% old accuracy.

The [single-pass stream comparison](evidence/full_reservoir_placement/README.md) also trains from the first task, using independently seeded class orders. Its three histories differ only in how the current-image group term is applied.

```text
python -B -s evidence/full_reservoir_placement/recompute.py --data evidence/full_reservoir_placement --output single_pass.json
```

## Data for training and new inference

No image download is needed for the CPU calculations above. For training or new inference, obtain the data from the original providers:

| Dataset | Source and local location |
| --- | --- |
| CIFAR-10 | The loaders use `torchvision.datasets.CIFAR10(..., download=True)` and save the official data under `data/r9_official/`. The [original dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) describes the official train/test split. |
| CIFAR-100 | Download the Python archive from the [original dataset page](https://www.cs.toronto.edu/~kriz/cifar.html). Keep it as `data/r9_external/cifar-100-python.tar.gz`. The preparation command below downloads it and prepares the training files. |
| CIFAR-10.1 | Download `cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy` from the [authors' repository](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets), and place both files in `data/r9_cifar101/`. |
| CIFAR-10.2 | Download `cifar102_test.npz` from the [authors' repository at the version used here](https://github.com/modestyachts/cifar-10.2/tree/521467a52ad494b78ff9c70051099c0d9ba74d40), and place it in `data/r13_cifar102/`. The preparation script below can obtain this file. |

After installing the training dependencies, prepare CIFAR-100 with:

```powershell
python -B -s -X utf8 -c "from experiments.r9_external.data import prepare_train_files; prepare_train_files('data/r9_external', download=True)"
```

Once CIFAR-10 is present and the CIFAR-10.1 files have been placed as above, this command downloads the missing CIFAR-10.2 test file and checks the external data and exact image overlap:

```powershell
python -B -s -X utf8 tools/prepare_r13_external_data.py --download-missing
```

It checks the complete local inputs, so obtain CIFAR-10 and CIFAR-10.1 first. Original images from CIFAR-10.1 and CIFAR-10.2 are not redistributed in this package. Configurations, sample selections, and numerical evidence are included. Third-party source records and licenses are under `third_party/`.

## Train the networks again

The recorded training environment used Windows, Python 3.12.9, PyTorch 2.11.0 with CUDA 12.6, and torchvision 0.26.0. `requirements-r9-windows-lock.txt` records the exact installed packages, including their wheel variants. In a separate Python 3.12 environment, the recorded Windows dependencies can be requested with:

```powershell
python -m pip install -r requirements-r9-windows-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://download.pytorch.org/whl/cpu
```

An NVIDIA GPU and a compatible driver are needed for the supplied training settings. Other operating systems require the corresponding PyTorch wheels; the Windows package list records the original environment. The commands below show one seed per comparison. Use a new output directory for each run, and use the other supplied configurations for the remaining seeds.

```powershell
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_followup/split_cifar10_classil_er_s201.json --data-root data --output retraining/split_cifar10_classil_er_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_followup/split_cifar10_classil_sgd_s201.json --data-root data --output retraining/split_cifar10_classil_sgd_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_derpp_challenge/split_cifar10_classil_derpp_s201.json --data-root data --output retraining/derpp_s201
python -B -s -X utf8 -m experiments.r18_cbp_official.engine --config configs/r18_cbp_official/nature_cifar_s201.json --data-root data --output retraining/cbp_s201
python -B -s -X utf8 -m experiments.r18_architecture_bridge.run --seed 201 --config experiments/r18_architecture_bridge/config.json --data-root data --output retraining/resnet32_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config evidence/history_controls/full_history16/er_s201/config.json --data-root data --output retraining/history16/er_s201
python -B -s -X utf8 tools/run_seen_classes.py --project . --method er --seed 201 --output retraining/seen_classes_er_s201
```

The full-training CIFAR-100 command runs the complete ten-task, five-seed comparison for SGD and ER, including the separate current-task learning comparisons:

```powershell
python -B -s -X utf8 tools/run_r15_normal_budget.py --run --manifest configs/r15_normal_budget/manifest.json --output retraining/cifar100
```

It uses 200 epochs on the first task and 70 on later tasks and the separate current-task comparisons. It takes substantially longer than the CPU analyses. The original comparisons reuse completed training. PAA preparation additionally trained the classifier-initialization, asymmetric-loss, and cross-group supervision comparisons documented below. Scripts that evaluate previously trained states require those states or newly trained replacements. Historical directory and argument names are retained so the original code remains runnable.

## Additional training comparisons

The entry points under `experiments/paa/` accept input and output locations. Keep original results as inputs and use a different output directory for each new run. All commands below run from this code directory. Obtain the official image files first. The full histories are produced by the preceding training commands; saved parameter states are needed for comparisons that start from those histories.

```powershell
python -B -s -X utf8 experiments/paa/asymmetric_loss/train.py --artifact-root . --data-root data --existing-results retraining --output-root retraining/asymmetric_loss
python -B -s -X utf8 experiments/paa/asymmetric_transfer_c100/train.py --data-root data/r9_external --existing-results retraining/cifar100 --output-root retraining/cifar100_incoming_loss
```

The first command expects the original five-seed SGD/ER runs in `retraining` with their supplied configuration names, for example `split_cifar10_classil_er_s201`. It trains two complete five-task loss histories per seed. The second starts from `jobs/seed3101_er` through `seed3105_er` in the completed CIFAR-100 campaign and trains only task 9. Both retain a 55-minute limit for the supplied settings. Measured preparation times were 7.1 minutes and 15.4 minutes, respectively, on an RTX 4060 Laptop GPU.

The cross-group supervision comparison starts from the same original CIFAR-100 campaign and uses a portable entry with the original training engine and recorded source files beside it:

```powershell
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py retraining/cifar100 data/r9_external retraining/cross_group_supervision --job mixed --check-inputs-only
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py retraining/cifar100 data/r9_external retraining/cross_group_supervision --job mixed
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py retraining/cifar100 data/r9_external retraining/cross_group_supervision --job blend
```

Run the two GPU jobs sequentially. The first trains ten models, the second five, each for 70 epochs with seeds 3101-3105. Each job has a 55-minute cap; the recorded times were 25.71 and 13.94 minutes. The entry performs training only and never loads test images. The CPU input check needs the original five saved pre-task states, but no image or plan document is loaded. See `experiments/paa/cross_group_supervision/README.md` for input layout and environment details.

For the classifier-initialization comparison, prepare the saved four-epoch and sixteen-epoch histories, then run these stages in order:

```powershell
python -B -s -X utf8 experiments/paa/head_state/train.py --stage prepare --project-root . --history4-root retraining --history16-root retraining/history16 --data-root data --output-root retraining/classifier_initialization
python -B -s -X utf8 experiments/paa/head_state/train.py --stage pilot --project-root . --history4-root retraining --history16-root retraining/history16 --data-root data --output-root retraining/classifier_initialization
python -B -s -X utf8 experiments/paa/head_state/train.py --stage run --project-root . --history4-root retraining --history16-root retraining/history16 --data-root data --output-root retraining/classifier_initialization
python -B -s -X utf8 experiments/paa/head_state/train.py --stage summarize --project-root . --history4-root retraining --history16-root retraining/history16 --data-root data --output-root retraining/classifier_initialization
```

The four-epoch history directory contains `split_cifar10_classil_er_s201` and corresponding method/seed directories. The sixteen-epoch directory contains `er_s201`, `sgd_s201`, and the other seeds 202–205. Each history must retain its pre-task models and original initialization. This comparison restores only the classifier parameters, then updates the entire model with the original current-only SGD procedure. The 80 new branches took 3.5 minutes. Training entry points use the bundled original experiment implementations; their comments and historical argument names are retained.

## Compare the timing of supervision

- [CIFAR-10 timing](evidence/reservoir_supervision_timing/README.md) compares identical new-image supervision at first presentation and at retrieval, plus an expected-count rule. It includes CPU commands for old, new, and overall accuracy during each task and at its endpoint.
- [Twenty-task CIFAR-100 timing](evidence/er_ace_timing_c100/README.md) gives paired residual-network placement runs, old-task forgetting, matched retention, and a separate validation-offset comparison.
- [Fashion-MNIST placement](evidence/fashion_placement/README.md) provides a second set of whole-stream training runs, the original data links, and the commands for the two measured-retention comparisons.
- `python -B -s tools/compare_same_candidates.py --artifact . --output same_candidates.json` compares the same two or thirty models with and without fitted offsets.


## Code availability

The accompanying files include experiment code, numerical evidence, original data-source links, dependency versions, and the commands above. Original image datasets remain subject to their providers' terms. Third-party licenses are retained with the corresponding source files. The publication repository will be released after author approval.

The additional fixed-margin comparison can be recomputed from the same inputs:

```powershell
python -B -X utf8 tools/compare_fixed_margin.py --artifact . --data evidence/score_frontier --output fixed_margin_reader
```

It uses a fixed five-percentage-point validation margin for the 30% old-class target, retains unavailable seeds, and applies the same validation choices on both test sources. It is a descriptive comparison, with no population guarantee.

## Recreate the article figures

The `plotting/` directory supplies the recorded figure inputs, plotting scripts, editable vectors, and numerical table scripts. Its README lists the commands and separate plotting dependencies. The listed commands have been run from that directory without relying on the original research checkout.
