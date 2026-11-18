# Complete Fashion-MNIST streams

The experiment compares five placements of the group term across ten complete five-task streams. A dataset of grayscale clothing images and a fully connected network complement the convolutional CIFAR runs. Each policy receives the same training images and retrieved examples. The main paired summary averages the four later tasks within each seed; the matching analysis retains tasks that can attain the specified old accuracy under both policies.

From the repository root, the 400 per-image files and training weights reproduce the recorded test summaries with NumPy:

```text
python -B -s evidence/reservoir_supervision_timing/recompute.py --data evidence/fashion_placement/measurements --old-target 30 --output fashion30.json
python -B -s evidence/reservoir_supervision_timing/recompute.py --data evidence/fashion_placement/measurements --old-target 80 --output fashion80.json
```

For new training, obtain the four original `.gz` dataset files from the [dataset authors](https://github.com/zalandoresearch/fashion-mnist) or [TensorFlow's copy of the original files](https://www.tensorflow.org/datasets/catalog/fashion_mnist). Keep them compressed in the same directory. The contents are not included here. Use the CUDA training dependencies listed in the repository README:

```text
python -B -s -u evidence/fashion_placement/train_fashion.py --data-root DIRECTORY_WITH_GZIP_FILES --output NEW_OUTPUT
```

To compare both timing rules from a common model before every task, add `--shared-history` to the training command. For the supplied outputs, use `shared_measurements` in the analysis command. Use `--old-target 70` for the new comparison. The `frontiers.py` and `calibrate_timing.py` programs in the companion CIFAR-10 directory also accept both Fashion-MNIST measurement directories.
