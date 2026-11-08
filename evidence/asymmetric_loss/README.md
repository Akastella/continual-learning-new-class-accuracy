# Portable CIFAR-10 predictions and counts

This directory needs Python and NumPy only. It contains all200 learned endpoints from the four-method, five-seed, five-task comparison, with predictions on the current task and every earlier task. It includes the common-current-only models' old-task predictions. No model files are needed.

Run `python recompute.py` to reproduce600 integer-count rows and five-seed statistics, including old-class accuracy for both usual and common-SGD endpoints. It verifies the retained counts and every previously reported mean and sample SD. `original_statistics.json` is the statistics file from the completed experiment; `recomputed_summary.json` adds the common-SGD old-class evaluations.

Each NPZ is named by seed, method, training task and endpoint. `predictions_all10` uses all ten candidates. `predictions_task2` restricts each test image to the TWO classes of its evaluation task, recorded by `evaluation_task`. The original paper comparison uses the current task's subset for this score. `mapped_labels` are global output indices0–9; `class_order[mapped_labels]` gives original CIFAR-10 labels; `official_test_indices` identifies rows in the official test split. Arrays are ordered by evaluation task, then the retained official test order. No pickle or PyTorch import is used.

The five independent seeds are201–205. Primary means first average post-initial tasks1–4 within each seed. Old-class accuracy at an endpoint averages all earlier-task examples, then these endpoint accuracies are averaged within seed. `counts.csv` gives integer counts for every endpoint/evaluation-task pair, including task0.
