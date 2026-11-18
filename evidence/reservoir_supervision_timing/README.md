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
