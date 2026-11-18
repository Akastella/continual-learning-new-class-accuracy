# ciFAIR-100 replacement images

The 20 prediction files describe two old-only replay losses at tasks eight and ten of five CIFAR-100 histories. Only the 927 replacements in ciFAIR-100 v1.0 are used, limited to the seen classes within each seed. The offsets come from the original CIFAR-100 validation fit at the 20% old target and a three-point margin. Paired changes in new accuracy and test old-target misses use those offsets without any fitting on ciFAIR.

```text
python -B -s recompute.py --output recomputed_cifair
```

Python and NumPy suffice for this command. It produces integer counts, seed scores, paired effects, and old-task outcomes from the supplied per-image scores. `replacement_rows.csv` identifies the replacement rows and their original class labels.

To repeat inference, download the [ciFAIR-100 v1.0 test file](https://raw.githubusercontent.com/cvjena/cifair/cf4bb3892429c34711c5c9f43aab6c32c75b7ca1/cifair-100/test) from the [dataset authors](https://cvjena.github.io/cifair/). Use the repository's PyTorch environment and matching model state.

```text
python -B -s predict.py CIFAIR_TEST CHECKPOINT scores.npz --task-index 7 --seed 3101 --condition full
```

Use task index 7 or 9, a seed from 3101 to 3105, and the corresponding `full` or `restricted` endpoint. See the eighth-task and final-task CIFAR-100 training instructions for generating those model states. The original dataset citation is [Barz and Denzler (2020)](https://doi.org/10.3390/jimaging6060041).
