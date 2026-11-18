# Supervision placement across a complete single-pass stream

This experiment trains three continual-learning histories per seed, with one pass through each CIFAR-10 task and ten new seeds, 4101–4110. Every seed has the same current and replay exposures across all three losses. The group term for newly arriving classes is placed on either current images retrieved from the reservoir, the scheduled current images at the same summed weight, or neither group. Old replay uses the all-seen loss throughout.

From the repository root:

```text
python -B -s evidence/full_reservoir_placement/recompute.py --data evidence/full_reservoir_placement --output single_pass.json
python -B -s -u evidence/full_reservoir_placement/train.py --data-root DATA_ROOT --output NEW_OUTPUT
```

The first command requires only NumPy. It reads the supplied training counts and 150 per-task score files, reproduces all reported means, paired differences and criteria, and writes a new JSON file. The second requires the CIFAR-10 PyTorch/CUDA environment described in the repository README. It uses the original CNN and reservoir implementation, requires only the original dataset, and prepares the predictions and final models. To recompute from a new run, pass that run directory to `--data`.
