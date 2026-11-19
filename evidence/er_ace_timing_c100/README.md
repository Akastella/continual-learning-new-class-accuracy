# Supervision timing in a twenty-task CIFAR-100 replay stream

This dataset has 20 tasks in the original five-consecutive-class order. The reference learner uses the reduced ResNet-18, cosine classifier, arriving-output mask and separate supervised means used in the ER-ACE experiment. The runs here set the default SGD rate to 0.1 and use no augmentation. See [the published paper](https://openreview.net/pdf?id=N8MaByOzUfb) and [its public implementation](https://github.com/pclucas14/AML/blob/paper_open_source/methods/er_ace.py) for the original method.

Each training condition shares the images, order and memory draws. The numerical files contain the predictions, old and new group scores, labels and applied weight at updates 25 and 250. Initial-task predictions for the forgetting matrices are stored once per seed because all placements have the same loss there. No model files or images are included.

From the root of the code package, recompute with the NumPy dependency:

```text
python -B -s evidence/er_ace_timing_c100/summarize_c100.py --data evidence/er_ace_timing_c100/measurements --first-task evidence/er_ace_timing_c100/first_task --output c100_timing.json
python -B -s evidence/reservoir_supervision_timing/calibrate_timing.py --data evidence/er_ace_timing_c100/measurements --old-target 20 --output c100_calibrated.json
```

The first command gives the early and mean end-of-task results, separate last-task results, 10%, 20% and 30% matched accuracy, coverage, and earlier-task forgetting. The second fits each score offset on 20 evaluation images per class and tests on the other 80.

To retrain, use the GPU requirements in the repository README and prepare the official CIFAR-100 train and test data. In this example, DATA_ROOT is the parent directory of the extracted cifar-100-python directory.

```text
python -B -s -u evidence/er_ace_timing_c100/train_c100.py --data-root DATA_ROOT --output NEW_RUN
python -B -s -u evidence/er_ace_timing_c100/train_c100.py --data-root DATA_ROOT --first-task-only --output NEW_FIRST_TASK
```

Use `--capacity 2000 --policies retrieval first_presentation` for the smaller memory run, and `--capacity 2000 --first-task-only` for its initial task predictions. Recompute the supplied smaller experiment with `smaller_measurements` and `smaller_first_task`.

The summaries also give the raw and paired new-class accuracy for every
later task, together with the paired changes and coverage at 20% measured
old-class accuracy. The same seed-task pair must be available at updates
25 and 250 for the matched comparison.
