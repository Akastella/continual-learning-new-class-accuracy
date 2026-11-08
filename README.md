# Code and statistical materials

This package accompanies the manuscript on interpreting new-class accuracy in class-incremental learning. It contains the original experiment code, configurations, and saved numerical evidence. The reported statistics can be recomputed on a CPU without downloading images or installing PyTorch. Training the networks again is a separate, substantially longer GPU task.

## Recompute the retention-target score comparison

The added journal comparison uses the same validation images and existing networks. Its complete numerical inputs include all 300 validation candidates and 600 test outputs across CIFAR-10.1 and CIFAR-10.2. From this directory, run:

```powershell
python -B -X utf8 tools/recompute_score_frontier.py --artifact . --data evidence/score_frontier --output recomputed_score_reader
```

This NumPy-only command recomputes scalar choices, all selected old/current/all accuracies, test violations, and the old-class accuracy bounds. It does not require the original research directory, images, checkpoints, or PyTorch. The chosen scalar values were obtained on validation images; test images are only scored. Complete test curves describe the stored models and are not used to choose scalar values.

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

The output directory must not already exist. A completed run is supplied in `recomputed_paa/`; the example uses a different name to preserve that run. For an isolated environment, create a virtual environment first and use its Python executable for both commands.

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

The package includes integer counts and saved per-example predictions, labels, official sample indices, and task identifiers where used by these analyses. Compact prediction files retain both the original argmax and the prediction among current classes. Some external-test analyses also include saved logits. These commands recompute statistics from saved evidence; they do not rerun network inference or training. Full model states and all original logits are retained in the research workspace rather than included in this compact package.

The PAA preparation run completed all eleven commands on 7 November 2026 with Python 3.12.9 and NumPy 1.26.4. The log timestamps span approximately 24 seconds. See `recomputed_paa/RUN_REPORT.md` for the command, environment, and output counts.

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
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_followup/split_cifar10_classil_er_s201.json --data-root data --output retraining/er_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_followup/split_cifar10_classil_sgd_s201.json --data-root data --output retraining/sgd_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config configs/r9_derpp_challenge/split_cifar10_classil_derpp_s201.json --data-root data --output retraining/derpp_s201
python -B -s -X utf8 -m experiments.r18_cbp_official.engine --config configs/r18_cbp_official/nature_cifar_s201.json --data-root data --output retraining/cbp_s201
python -B -s -X utf8 -m experiments.r18_architecture_bridge.run --seed 201 --config experiments/r18_architecture_bridge/config.json --data-root data --output retraining/resnet32_s201
python -B -s -X utf8 -m experiments.r9_followup.engine --config evidence/history_controls/full_history16/er_s201/config.json --data-root data --output retraining/full_history16_er_s201
python -B -s -X utf8 tools/run_seen_classes.py --project . --method er --seed 201 --output retraining/seen_classes_er_s201
```

The full-training CIFAR-100 command runs the complete ten-task, five-seed comparison for SGD and ER, including the separate current-task learning comparisons:

```powershell
python -B -s -X utf8 tools/run_r15_normal_budget.py --run --manifest configs/r15_normal_budget/manifest.json --output retraining/cifar100
```

It uses 200 epochs on the first task and 70 on later tasks and the separate current-task comparisons. It takes substantially longer than the CPU analyses. The CPU results use completed training runs; no networks were trained during PAA preparation. Scripts that evaluate previously trained states require those states or newly trained replacements. Historical directory and argument names are retained so the original code remains runnable.

## Code availability

**Author to complete:** provide the public repository or permanent archive URL for this study's code and statistical materials. No public URL for this study was found in the supplied package. The dataset and third-party links above identify their respective providers. This local preparation did not publish or upload the package.
