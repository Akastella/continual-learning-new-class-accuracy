# Prospective CIFAR-100 final-task loss intervention

Written after the completed CIFAR-10 asymmetric-loss comparison and before any new CIFAR-100 training output or outcome inspection. The CIFAR-10 result motivates the following directional transfer prediction; it does not select this experiment from new CIFAR-100 outcomes.

## Fixed history and single changed factor

Use the completed campaign at E:/Projects/iclr 27/results/r15_normal_budget_recovery1. The earlier r15_normal_budget_v1 attempt contains only part of the matrix. For seeds3101–3105, start from jobs/seedSEED_er/checkpoints/pre_task_09.pt, which already includes the original reset of new classifier rows. Reuse the corresponding actual_task_09.pt/evaluation/actual_task_09.npz baseline and unchanged common-current-only probe_task_09.pt/evaluation/probe_task_09.npz. No history task is retrained and no common-current-only model is retrained.

Use the original archived source in recovery1/source, verified against its manifest: ResNet32 with GroupNorm8 and linear100-output head; first history task200epochs and subsequent history tasks70epochs; final task70epochs; SGD lr0.1, momentum0.9, no Nesterov, weight decay0.0002, LR multipliers0.1 after epochs30 and50; BF16 training, FP32 parameters and evaluation, channels-last, deterministic execution and TF32 disabled; current64 and replay64 except the paired eight-example final batches; memory20 per each of90 old classes, original fixed one-to-one replay plan. Use the existing official CIFAR-100 train archive and original class permutations.

Only the current images' candidate classes change from all100 outputs to the current10 outputs90–99. Replay images keep all100 candidates. Keep the concatenated current-then-replay forward and the original single cross-entropy mean over the full concatenated batch, applying -1e9 only to the old90 outputs on current rows. This retains identical current/replay weights and total loss scale, as well as batch shape. The original train_branch function remains unchanged and receives this loss through its local functional interface.

The original augment function reconstructs a local CUDA generator from seed/task/epoch/batch/source for zero-pad4, crop32, flip0.5, continuous brightness63/255 and normalization. It consumes no global RNG. Optimizer and scheduler start anew for each branch. Thus the saved pre-task parameters and deterministic event keys fully determine the intervention; no unsaved global augmentation or momentum state is needed.

## Predictions and decision rules

Relative to the unchanged final-task ER baseline, predict (1) increased competition loss, defined as current10-restricted correct minus all100-candidate correct on the SAME1000 new-class test images, and (2) decreased usual new-class all100-candidate accuracy. The transfer prediction is supported only when the paired seed mean of the competition-loss difference is positive with a strict majority of seeds positive, AND the paired seed mean of the usual-new-accuracy difference is negative with a strict majority of seeds negative. Zero is not a directional success. Report every seed and every counterexample; failure of either condition is reported as failure of this transfer prediction, not redefined as success.

Also predict increased old-class all100-candidate accuracy and report it separately; it is not a substitute for the two necessary directions. Do not require the within-current10 accuracy change to be zero or small. Report current1000, old9000 and all10000 test correct/total counts, current10-restricted correct, all per-example predictions and the unchanged common-current-only outputs. Statistical unit is the paired seed (n=5); no averaging over multiple tasks is involved.

This is a fixed-history final-task loss intervention, distinct from the CIFAR-10 experiment that changed the loss throughout each history. It transfers an asymmetric candidate restriction to this protocol; it does not reproduce the original ER-ACE architecture or benchmark. CIFAR-100 was previously evaluated in the project; the newly specified loss intervention and directional test are prospective.

## Verification, resources and preservation

Verify all archived source files, completed campaign status, pre-state identity, baseline endpoint and NPZ identities. Recompute all70epoch current/replay/augmentation identities and compare them to the retained baseline. Run one necessary small unmasked training comparison from the same pre-state to check that the local loss interface reproduces the original train_branch update; do not retrain the complete baseline. All five intervention endpoints are trained before new scientific test evaluation.

Original final-task70epoch ER took133–143seconds per seed on this machine (5seed sum676.6seconds). Expected current complete job15–25minutes including input checks, one-epoch operator verification, persistence and final evaluation; one sequential GPU job with55minute wall-clock cap. A train-only timing check can refine this estimate before the full matrix. If the projected completion exceeds55minutes, stop and report the evidence and a smaller valid design before changing the protocol. Preserve every partial log if a failure occurs; no outcome-dependent retries or exclusions. No new labels or packages.

Write everything only under this directory. Retain local model files for inspection, but the reader package consists of NumPy predictions, labels/indices, integer counts, original statistics and a standalone NumPy recomputation script; do not include large model files in that package or push them to Git.
