# The loss on current images retrieved from a reservoir

Two conditions per seed use the fifth task of ten CIFAR-10 histories. Each pair starts from the same original ER model and saved reservoir. Scheduled current images use a two-class loss, old replay uses all ten, and only the loss on retrieved current images differs. Uniform sampling and insertion are the original ER procedure. The prediction files include the group scores and predicted classes for 10,000 test images per endpoint.

Run these commands from the repository root:

```text
python -B -s evidence/reservoir_replay_loss/recompute.py --output reservoir.json
python -B -s -u evidence/reservoir_replay_loss/train.py --history HISTORY_ROOT --data-root DATA_ROOT --device cpu --output NEW_OUTPUT
```

The recomputation needs Python and NumPy. Training uses the CIFAR-10 PyTorch environment and the original saved fifth-task start files, including the reservoir. Prepare them with the original ER training commands in the repository README. Training creates the same layout of group scores in the specified directory. To recompute from a new run, pass `--data NEW_OUTPUT` and a new JSON destination to the first command.
