# Supervision placement across a complete single-pass stream

This experiment trains three continual-learning histories per seed, with one pass through each CIFAR-10 task and ten new seeds, 4101–4110. Every seed has the same current and replay exposures across all three losses. The group term for newly arriving classes is placed on either current images retrieved from the reservoir, the scheduled current images at the same summed weight, or neither group. Old replay uses the all-seen loss throughout.

From the repository root:

```text
python -B -s evidence/full_reservoir_placement/recompute.py --data evidence/full_reservoir_placement --output single_pass.json
python -B -s -u evidence/full_reservoir_placement/train.py --data-root DATA_ROOT --output NEW_OUTPUT
```

The first command requires only NumPy. It reads the supplied training counts and 150 per-task score files, reproduces all reported means, paired differences and criteria, and writes a new JSON file. The second requires the CIFAR-10 PyTorch/CUDA environment described in the repository README. It uses the original CNN and reservoir implementation, requires only the original dataset, and prepares the predictions and final models. To recompute from a new run, pass that run directory to `--data`.

## Class-balanced retrieval

The same ten CIFAR-10 streams also use batches drawn as evenly
as possible across resident classes. The underlying reservoir
storage and the three matched loss conditions are the same.
The balanced directory holds 150 prediction files and the
training counts. The matched-old results use the tasks for which
all three losses attain each specified old-class accuracy.

    python -B -s evidence/full_reservoir_placement/recompute.py --data evidence/full_reservoir_placement/balanced --old-targets 20 30 40 --output balanced_summary.json
    python -B -s -u evidence/full_reservoir_placement/train_balanced.py --data-root DATA_ROOT --output NEW_BALANCED_RUN

To fit score offsets at 33% old accuracy on validation and test them
against a 30% old-accuracy target on the other images, run:

    python -B -s evidence/full_reservoir_placement/recompute.py --data evidence/full_reservoir_placement/balanced --old-targets 20 30 40 --validation-old-target 30 --output balanced_calibrated.json

The output includes the common feasible task pairs, actual test scores,
and misses under each policy. The measured-score rows in the uniform
and balanced reports also list all eligible seed-task pairs. Their
intersection supports comparisons on identical tasks across retrieval
schemes.
