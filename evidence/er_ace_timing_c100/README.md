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

## Reorder the loss weights on random class streams

Five seeds (7201–7205) each use a different random ordering of the
100 labels. At each later task, three branches share the same model,
arrivals and replay images. The weight of the group loss on arriving
images follows the expected count, the same weights in reverse order,
or reverse order through update 187 followed by the original weights
at updates 188–250. The increasing branch continues the history.

The first command recomputes the original-score, measured-retention,
and validation-fitted results from the supplied predictions. The
second command reruns training using the original data.

    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/er_ace_timing_c100/order_measurements --output c100_order.json
    python -B -s -u evidence/er_ace_timing_c100/train_order_c100.py --data-root DATA_ROOT --output NEW_RUN

The measurements include 570 prediction and score files and the
class order, per-update coefficients and cumulative weight of each
branch. See the paper's supplement for the sampling and fitting rules.

## Changing only the last 63 weights

From the model after update 187 under the fully reversed schedule,
train each remaining minibatch with its original small weights
or with the 63 larger weights from the increasing schedule, in
either order. The increasing-weight history supplies task starts
as in the previous comparison. The data contain 285 paired
task-end score files and their full coefficient schedules.

    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/er_ace_timing_c100/tail_measurements --reference decreasing --additional-old-targets 10 30 --output tail_summary.json
    python -B -s -u evidence/er_ace_timing_c100/train_tail_c100.py --data-root DATA_ROOT --output NEW_TAIL_RUN

For the earlier weight-order experiment, add the option
--additional-old-targets 10 25 30 to the summary command to
produce the multi-target output stored in order_targets.json.


## Exchange equal-weight windows and compare fitted scores

The window-exchange run swaps the first and last 63 coefficients
of the fully reversed schedule and reuses its middle 124. The
increasing model supplies the common task starts. The stored
fully reversed comparator comes from order_measurements, and
the re-created increasing runs have identical stored outputs.

    python -B -s -u evidence/er_ace_timing_c100/train_order_c100.py --data-root DATA_ROOT --exchange-windows --output NEW_EXCHANGE_RUN
    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/er_ace_timing_c100/exchange_run --reference decreasing --extra-reference evidence/er_ace_timing_c100/order_measurements --additional-old-targets 10 30 --output exchange_summary.json
    python -B -s evidence/er_ace_timing_c100/compare_tail_offsets.py --data evidence/er_ace_timing_c100/tail_measurements --output tail_offset_summary.json

The last command fits a scalar offset to match the comparator's
old-group accuracy on the validation images and evaluates it on
separate images. Its output includes per-task feasibility, paired
test accuracies and individual-prediction agreement.

Adding --original-label-split to the order or exchange summary command
uses an alternative validation split based on the original class label,
so each image stays on the same side of the split across class orders.
Append --additional-old-targets 10 25 30 for the full order sensitivity.
The two reference outputs are original_identity_order.json and
original_identity_exchange.json. The training commands for the order
and exchange comparisons accept --seeds 7201 to run a single stream.

## Continue both schedules into the next task

At tasks 5, 10 and 15 in the same five histories, the descending and
exchanged schedules receive the same batches and increasing weights
on the next task. Predictions at 0, 25, 100 and 250 further updates
use the preceding task's set of seen classes. Zero denotes the
end of the original comparison.

    python -B -s -u evidence/er_ace_timing_c100/train_continuation_c100.py --data-root DATA_ROOT --output NEW_CONTINUATION_RUN
    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/er_ace_timing_c100/continuation_run --reference decreasing --additional-old-targets 10 30 --output continuation_summary.json

Training can be run one history at a time with the --seeds option.
The statistic command uses all five histories in the supplied data.
The outputs contain original-score and matched-accuracy measures,
paired differences at the four updates, and per-task coverage.

## Omit the group loss from the same CIFAR-100 task starts

At each of the 95 random-order task starts, the added branch sets
all arriving-batch group weights to zero. The original timing and
replay settings apply throughout. The published numerical inputs
provide the new omission predictions; the increasing and descending
branches are supplied in the original order directory.

    python -B -s -u evidence/er_ace_timing_c100/train_order_c100.py --data-root DATA_ROOT --omit-group-loss --output NEW_OMISSION_RUN
    python -B -s evidence/er_ace_timing_c100/summarize_order_c100.py --data evidence/er_ace_timing_c100/omission_run --reference increasing --extra-reference evidence/er_ace_timing_c100/order_measurements --policies increasing decreasing omit --additional-old-targets 10 25 30 --original-label-split --describe-groups --output omission_summary.json

The statistics place the three branches at the same measured old
accuracy, fit offsets on disjoint image identities, and compare the
score margins on images with the same correct within-group label.

## Compare with the reference's own old accuracy

Use the increasing model's original old accuracy as the target at each
task. The test-curve comparison uses test labels to locate the point.
The second comparison fits on validation images and reports the test
accuracy and any misses relative to the increasing model.

    python -B -s evidence/er_ace_timing_c100/compare_reference_offsets.py --reference evidence/er_ace_timing_c100/order_measurements --omission evidence/er_ace_timing_c100/omission_run --output reference_retention_c100.json

For CIFAR-10, use the order_measurements and order_omission_run
directories in evidence/reservoir_supervision_timing instead.
