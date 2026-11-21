# Timing of the group loss

Ten CIFAR-10 histories give a common model and reservoir at each of the four later tasks. Three branches use identical training exposures. The same new images receive an equal total weight when the group term is applied at retrieval or first presentation. A third placement weights the scheduled new images by the expected group-loss weight, computed from arrival counts.

Recompute on CPU from the repository root:

```text
python -B -s evidence/reservoir_supervision_timing/recompute.py --data evidence/reservoir_supervision_timing/measurements --old-target 20 --output timing.json
```

The supplied measurements include 480 files with old/new group scores, predictions, and labels at four updates, plus the actual applied weights. For matched old accuracy, the analysis uses a shared test-score offset giving the largest new accuracy at the stated old requirement. These are descriptive curves using test labels. Retrain on a GPU with the CIFAR-10 inputs prepared as described in the repository README:

```text
python -B -s -u evidence/reservoir_supervision_timing/train_timing.py --data-root DATA_ROOT --output NEW_OUTPUT
```

For complete follow-through with a separate training history for each timing rule, add `--carry-history` to the training command. Use `carried_measurements` to recompute the supplied results.

```text
python -B -s evidence/reservoir_supervision_timing/frontiers.py --data evidence/reservoir_supervision_timing/measurements --output timing_frontiers.json
python -B -s evidence/reservoir_supervision_timing/calibrate_timing.py --data evidence/reservoir_supervision_timing/measurements --old-target 20 --output timing_calibrated.json
```

The first calculation uses test labels to describe the region each model can reach, using the same available seed-task pairs at both measurements for each point. The second fits each offset on separate evaluation images and measures test accuracy and misses.

## Reorder the same arriving-batch weights

Seeds 5201–5210 use increasing expected-count weights, the same weights
reversed over the full minibatches, and retrieval. Every branch for a
later task starts from the common retrieval-trained model and reservoir.
The numerical inputs contain per-update coefficients, test predictions,
and group scores for all 40 paired tasks at the four recorded updates.

Run from the repository root:

```text
python -B -s evidence/reservoir_supervision_timing/recompute.py --data evidence/reservoir_supervision_timing/order_measurements --reference increasing --output order_summary.json
python -B -s evidence/reservoir_supervision_timing/frontiers.py --data evidence/reservoir_supervision_timing/order_measurements --policies increasing decreasing --output order_frontiers.json
python -B -s evidence/reservoir_supervision_timing/calibrate_timing.py --data evidence/reservoir_supervision_timing/order_measurements --old-target 20 --reference increasing --output order_calibrated.json
python -B -s -u evidence/reservoir_supervision_timing/train_order.py --data-root DATA_ROOT --output NEW_OUTPUT
```

The first three commands use the supplied predictions and require
only NumPy. They report original scores, the tradeoff at measured
old accuracy, and accuracy after fitting an offset on separate
validation images. The final command trains the classifiers on CUDA.

## Omit or reorder the same loss

The omission comparison uses the same seeds, task starts, and batches as
the order comparison above. It sets every arriving-batch group-loss
weight to zero. The repeated retrieval histories reproduce the original
recorded scores at all four updates. The three alternatives share the
same reference model at the start of each later task. The supplied
omission directory contains the 160 new score files. The retrieval
predictions are supplied once, in the original order directory.

    python -B -s -u evidence/reservoir_supervision_timing/train_order.py --data-root DATA_ROOT --omission-only --output NEW_OMISSION_RUN
    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/reservoir_supervision_timing/order_omission_run --reference increasing --extra-reference evidence/reservoir_supervision_timing/order_measurements --policies increasing decreasing omit --additional-old-targets 30 40 --output order_omission_summary.json

The second command uses NumPy alone. It joins the zero-weight branch
with the previously measured increasing and decreasing schedules. The
output gives common coverage and per-task and per-history results.
