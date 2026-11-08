# R18 compact architecture-bridge evidence

This integer-only package contains official CIFAR-10 indices, mapped and
original labels, argmax predictions for every reported endpoint, and exact
training/correct-count records. It excludes float logits, checkpoints, and
reservoir tensors. The complete raw evidence remains in
`results/r18_architecture_bridge_v1`.

Recompute all reported R18 endpoints and sign counts from a clean extraction:

```powershell
python -s -X utf8 recompute.py --root . --output recomputed.json
```

`recomputed.json` is intentionally excluded from the frozen inventory because
it is the command's fresh output. The evidence is descriptive: five paired
seeds on the previously used official CIFAR-10 test split.
