# Move the all-class loss to nonrepeated current images

The 10 prediction files are one new condition at the eighth and tenth CIFAR-100 tasks in five histories. The training images, order, augmentation, optimizer, and per-image weights are shared with the mixed condition described in [revisit supervision](../revisit_supervision/README.md). All-class supervision on 32 repeated current presentations is moved to the 32 scheduled current images that occur once in the update. Repeated current images use ten-class cross-entropy in this condition.

Run from the repository root:

```text
python -B -s evidence/shift_current_loss/recompute.py --artifact . --output relocated.json
python -B -s -u evidence/revisit_supervision/train.py PRE_TASK_07.pt DATA_ROOT NEW_OUTPUT --task-index 7 --seed 3101 --condition all_seen_on_nonrepeated
```

The first command needs NumPy and reads the existing mixed predictions alongside the ten added predictions. It reports raw, within-group, and validation-selected results. The second trains one endpoint with the shared CIFAR-100 engine. Use task index 7 or 9 and the matching original ER pre-task model.
