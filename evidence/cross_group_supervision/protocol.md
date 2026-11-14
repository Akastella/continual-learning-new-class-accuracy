# Current-image cross-group supervision on the fixed CIFAR-100 histories

Scientific design excerpt from the recorded training plan. The introductory administrative paragraph is omitted; the unabridged original remains in the authors' working directory. This excerpt is not byte-identical to that original plan. Execution records retain the original plan SHA-256 identifier. The original five paired ER pre-task-9 states, seeds 3101-3105, remain unchanged. The experiment tests current-image all-class supervision under the fixed conditions below.

## Comparisons and hypotheses

The existing two conditions are original ER and current-ten incoming loss, each with replay containing only old images. Add three prespecified conditions, all from the same pre-task states and all for 70 epochs:

1. Full-class incoming loss, with half of each replay batch replaced by current images.
2. Current-ten incoming loss, with the identical mixed replay batch.
3. Original old-only replay, with current-image loss 0.95 times current-ten CE plus 0.05 times all-hundred CE. Replay uses ordinary all-hundred CE.

For conditions 1–2, keep the original current batch and original replay batch sizes. In the replay batch retain the first half of the original old-image indices and replace its second half by the first half of the current batch's indices. Thus a full batch has 64 scheduled current, 32 old replay, and 32 revisited current examples; the final batch has 8/4/4. Current examples in the replay term receive the original replay augmentation keys. Both losses use this identical schedule. This deliberately changes current and old exposure relative to old-only replay; the two-by-two comparison is required to distinguish its effect from that of the incoming loss. No claim of equal old exposure across the replay policies is allowed. Both mixed conditions have identical exposures, updates, architecture, augmentation schedule, and loss scale.

Hypothesis H1: adding current images to the all-class replay term reduces the incoming-restriction penalty. Define the restriction effect as restricted-minus-full incoming accuracy within a replay policy. The mixed-minus-old-only change in this effect must be positive for unrestricted new accuracy and negative for competition loss, both in paired mean and a strict majority of seeds. Failure of either required direction rejects H1. Report the two endpoint scores and current-ten accuracy as well, rather than using this interaction alone.

Hypothesis H2: a 5% all-class component on the existing current images is sufficient for substantial recovery while retaining an old-class advantage. Its unrestricted new accuracy must reach at least half the original full-loss ER baseline in paired mean and at least 3/5 seeds; its old accuracy must exceed that baseline in paired mean and at least 3/5 seeds. Otherwise H2 fails. Do not tune the coefficient or change the recovery threshold after output inspection. Also report exact effects relative to the zero-component restriction.

These are conditional directional/sufficiency comparisons, with seeds as the independent units (n=5). They do not establish a minimum sufficient coefficient, a universally optimal memory policy, or a new best method. Report all seed results, including failures. All-class accuracy weights old and new images 0.9/0.1.

## Implementation and resource estimate

Reuse the unchanged archived training engine and the completed intervention's model/data loading and evaluation. Change only the replay-index construction or the current CE combination as specified. Preserve original sources and saved outputs. A small train-only timing and identity check precedes new matrices. Existing measurement: five 70-epoch runs plus evaluation took 921 seconds. Expected mixed-policy job about 31 minutes; conservative 1.5-fold estimate about 46 minutes, below a 55-minute cap. The 5% job is separate, expected about 15 minutes and conservatively 23 minutes. Run one GPU job at a time, save all partial outputs, and evaluate test images only after all 15 new models finish training. Do not inspect intermediate scientific test outcomes. No new labels, packages, data, or hyperparameter search.

Retain models and training records locally under this directory on C:. Package predictions, official image identities, integer counts, seed effects, and standalone NumPy analysis for readers. The original 15 endpoint prediction files supply the two old-only baselines and the unchanged current-only comparison.
