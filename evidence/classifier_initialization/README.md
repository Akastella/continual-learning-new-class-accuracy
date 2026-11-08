# Recompute the classifier-state comparison with NumPy

Run:

```text
python recompute.py
```

Only NumPy and the Python standard library are required. No parameter files, PyTorch, GPU, images, network access, or training are needed to reproduce the count-based comparisons in this package.

`manifest.json` describes 160 method/seed/task/condition combinations. `predictions.npz` contains exact prediction, label, official-index and evaluation-task arrays for 120 saved endpoints: 80 common-original-classifier endpoints and 40 original-classifier endpoints after four-epoch histories. The 40 original-classifier endpoints after sixteen-epoch histories have only the original saved current-task correct count. Their old and all-seen results remain missing because the four-epoch endpoint was not retained. No missing predictions are imputed.

Array names are `<endpoint_id>_prediction`, `_label`, `_index`, and `_eval_task`. Predictions and labels use uint8 and values 0--9, with the same global output mapping as the original experiment. Official CIFAR-10 test indices use uint16 and values 0--9999. Evaluation-task IDs use uint8. Every evaluated model predicts among **all ten global classes**, including classes not yet seen; candidates are never restricted to the tested class pair. Each endpoint's arrays concatenate official test examples from task 0 through its current task, retaining within-task official index order.

For current task ID `t`, current examples have `eval_task == t`; old examples have `eval_task < t`. There are 2,000 current images and `2,000 * t` old images, with the old classes pooled within the same endpoint. All-seen accuracy is the total correct count divided by `2,000 * (t + 1)`. The current predictions preserve the exact original saved GPU outputs. New old-class predictions were evaluated from the saved final network using CPU only and four threads. Metadata records the evaluation source; the full study directory also records CPU/GPU agreement checks on all 160,000 new current predictions. Original-classifier history-4 predictions are copied from the existing original evaluation arrays after checking their network identifiers and counts.

The entry point validates array types, lengths, label-to-task mappings, uniqueness of official indices, and expected integer counts, then writes:

- `recomputed_endpoint_counts.csv`: 160 rows of old/current/all-seen correct counts, denominators and accuracies, with explicit missing original-history16 old scores.
- `recomputed_seed_scores.csv`: within-seed means over task IDs 1--4 and separately labelled sums of model-specific prediction counts. Those summed counts contain predictions from four distinct endpoints and should not be treated as one model's test accuracy.
- `recomputed_paired_differences.csv`: every seed's ER-minus-SGD difference for each classifier condition and outcome.
- `recomputed_summary.json`: across-seed means and sample SD, paired method differences, and common-classifier minus original-classifier differences in those method differences. Scores for task ID 4 are also supplied separately; they concern one endpoint tested on all 10,000 official images.

The primary seed summary first averages four endpoint-level task accuracies within each seed, then computes the mean and sample SD across seeds 201--205. The independent analysis unit is seed. Original sixteen-epoch old scores are missing throughout; they are never replaced with scores from an endpoint after sixteen current-task epochs.

The actual training runner and its unchanged original implementation are outside this compact NumPy analysis package. They can use author-provided original checkpoint parameters through the runner's input-location arguments. Complete `final.pt` parameter files are not included here.
