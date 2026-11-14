# Cross-group supervision training

This entry trains the three added final-task conditions from the completed original CIFAR-100 ER pre-task-9 states. Fifteen models use seeds 3101-3105 and 70 epochs each (1,050 epochs total). The actual recorded experiment has completed; these commands are for repeating its training.

Run from the code-package root. Install the recorded Windows/CUDA dependencies in `source/requirements-r9-windows-lock.txt`, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and use Python user-site isolation as below. On Windows, TEMP and TMP must point to a writable directory with sufficient space.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py HISTORY_ROOT DATA_ROOT NEW_OUTPUT --job mixed --check-inputs-only
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py HISTORY_ROOT DATA_ROOT NEW_OUTPUT --job mixed
python -B -s -u -X utf8 experiments/paa/cross_group_supervision/train.py HISTORY_ROOT DATA_ROOT NEW_OUTPUT --job blend
```

Replace the three positional locations with your input and output paths. The two GPU commands run sequentially in the same new campaign directory, with separate job markers and a 55-minute limit per job. `mixed` trains ten models (full and restricted incoming loss with identical mixed replay); `blend` trains five (5% all-class current CE with old-only replay). The actual mixed and blend jobs took 25.71 and 13.94 minutes on the recorded RTX 4060 Laptop GPU.

HISTORY_ROOT contains completed original `batch.json`, `manifest.json`, and `jobs/seed3101_er` through `jobs/seed3105_er`, each with its original `training_complete.json` and `checkpoints/pre_task_09.pt`. The full-history command in the package README produces this campaign. DATA_ROOT contains the official `cifar-100-python.tar.gz` and extracted `cifar-100-python/train` and `meta`. Downloads are disabled. NEW_OUTPUT needs no preplaced files. `--check-inputs-only` verifies the bundled source files and all five starting model states on a CPU; it performs neither training nor test loading.

`train.py`, `portable_math.py`, `original_manifest.json`, and `source/` are copied unchanged from the verified portable entry. The archived engine is unchanged. `portable_math.py` contains the same executed `mixed_plan`, `Loss`, and `train_one` operators; the supplied verification JSON records their equality. This training entry loads no plan document and never loads the official test payload. It saves completed endpoint states and epoch records. Models, full logits, and image datasets are not redistributed here. The original source files and 1,050 epoch records are in `evidence/cross_group_supervision/execution/`; saved small NumPy inputs and CPU commands are documented in that evidence directory.
