# Cross-group current-image supervision

All 15 new 70-epoch models completed from the unchanged pre-task-9 states before any new test evaluation. Five paired seeds (3101–3105) are the independent units. All accuracies are percentages; effects and competition loss are percentage points. Values show mean ± sample SD.

| Endpoint | New all100 | New current10 | Competition loss | Old all100 | All classes |
|---|---:|---:|---:|---:|---:|
| Original ER, old-only replay | 69.200 ± 2.680 | 73.400 ± 2.169 | 4.200 ± 0.828 | 14.924 ± 1.128 | 20.352 ± 0.780 |
| Restricted incoming, old-only replay | 0.720 ± 0.327 | 74.620 ± 3.046 | 73.900 ± 3.081 | 25.920 ± 1.063 | 23.400 ± 0.986 |
| Full incoming, mixed replay | 72.320 ± 2.184 | 75.640 ± 2.396 | 3.320 ± 0.779 | 15.049 ± 1.147 | 20.776 ± 0.839 |
| Restricted incoming, mixed replay | 69.740 ± 2.828 | 76.300 ± 2.493 | 6.560 ± 0.999 | 17.380 ± 1.428 | 22.616 ± 1.091 |
| 5% all-class current CE, old-only replay | 53.360 ± 3.120 | 74.500 ± 2.148 | 21.140 ± 2.673 | 22.140 ± 1.559 | 25.262 ± 1.293 |
| Unchanged current-only comparison | 76.540 ± 3.391 | 76.540 ± 3.391 | 0.000 ± 0.000 | 0.000 ± 0.000 | 7.654 ± 0.339 |

## Prespecified H1

H1 is **supported** by the fixed joint mean-and-majority criterion. The interaction is (mixed restricted − mixed full) − (old-only restricted − old-only full).

| Interaction endpoint | Paired mean ± SD | Positive / zero / negative seeds |
|---|---:|---:|
| new_all100_pct | 65.900 ± 3.374 | 5 / 0 / 0 |
| new_current10_pct | -0.560 ± 1.214 | 1 / 0 / 4 |
| competition_loss_pp | -66.460 ± 4.036 | 0 / 0 / 5 |
| old_all100_pct | -8.664 ± 0.817 | 0 / 0 / 5 |
| all100_pct | -1.208 ± 0.446 | 0 / 0 / 5 |

## Prespecified H2

H2 is **supported**. New all100 accuracy minus half the original ER accuracy is 18.760 ± 1.853 pp; 5/5 seeds meet that threshold. Old-class accuracy minus original ER is 7.216 ± 0.930 pp; 5/5 seeds improve. The coefficient remains 0.05.

| Blend05 minus zero-component restriction | Paired mean ± SD | Positive / zero / negative seeds |
|---|---:|---:|
| new_all100_pct | 52.640 ± 3.153 | 5 / 0 / 0 |
| new_current10_pct | -0.120 ± 1.390 | 2 / 0 / 3 |
| competition_loss_pp | -52.760 ± 4.530 | 0 / 0 / 5 |
| old_all100_pct | -3.780 ± 0.660 | 0 / 0 / 5 |
| all100_pct | 1.862 ± 0.439 | 5 / 0 / 0 |

| Seed | H1 new interaction | H1 competition interaction | H2 new minus half ER | H2 old minus ER |
|---|---:|---:|---:|---:|
| 3101 | +67.700 | -69.600 | +18.550 | +7.511 |
| 3102 | +69.700 | -70.200 | +21.300 | +8.422 |
| 3103 | +62.800 | -61.400 | +16.200 | +7.544 |
| 3104 | +67.400 | -68.200 | +19.450 | +6.044 |
| 3105 | +61.900 | -62.900 | +18.300 | +6.556 |

## Every seed and endpoint

| Seed | Endpoint | New all100 | New current10 | Competition loss | Old all100 | All classes |
|---|---|---:|---:|---:|---:|---:|
| 3101 | Original ER, old-only replay | 70.500 | 75.000 | 4.500 | 14.333 | 19.950 |
| 3101 | Restricted incoming, old-only replay | 0.600 | 77.200 | 76.600 | 25.311 | 22.840 |
| 3101 | Full incoming, mixed replay | 73.000 | 77.200 | 4.200 | 14.989 | 20.790 |
| 3101 | Restricted incoming, mixed replay | 70.800 | 77.500 | 6.700 | 16.544 | 21.970 |
| 3101 | 5% all-class current CE, old-only replay | 53.800 | 76.800 | 23.000 | 21.844 | 25.040 |
| 3101 | Unchanged current-only comparison | 78.600 | 78.600 | 0.000 | 0.000 | 7.860 |
| 3102 | Original ER, old-only replay | 72.000 | 75.400 | 3.400 | 14.444 | 20.200 |
| 3102 | Restricted incoming, old-only replay | 1.100 | 77.300 | 76.200 | 26.767 | 24.200 |
| 3102 | Full incoming, mixed replay | 74.400 | 76.700 | 2.300 | 14.333 | 20.340 |
| 3102 | Restricted incoming, mixed replay | 73.200 | 78.100 | 4.900 | 17.156 | 22.760 |
| 3102 | 5% all-class current CE, old-only replay | 57.300 | 75.500 | 18.200 | 22.867 | 26.310 |
| 3102 | Unchanged current-only comparison | 77.500 | 77.500 | 0.000 | 0.000 | 7.750 |
| 3103 | Original ER, old-only replay | 65.600 | 71.100 | 5.500 | 16.333 | 21.260 |
| 3103 | Restricted incoming, old-only replay | 1.000 | 71.700 | 70.700 | 26.756 | 24.180 |
| 3103 | Full incoming, mixed replay | 69.200 | 72.300 | 3.100 | 16.211 | 21.510 |
| 3103 | Restricted incoming, mixed replay | 67.400 | 74.300 | 6.900 | 18.989 | 23.830 |
| 3103 | 5% all-class current CE, old-only replay | 49.000 | 73.700 | 24.700 | 23.878 | 26.390 |
| 3103 | Unchanged current-only comparison | 74.800 | 74.800 | 0.000 | 0.000 | 7.480 |
| 3104 | Original ER, old-only replay | 70.700 | 74.500 | 3.800 | 13.644 | 19.350 |
| 3104 | Restricted incoming, old-only replay | 0.300 | 75.900 | 75.600 | 24.344 | 21.940 |
| 3104 | Full incoming, mixed replay | 74.000 | 78.000 | 4.000 | 13.567 | 19.610 |
| 3104 | Restricted incoming, mixed replay | 71.000 | 78.600 | 7.600 | 15.578 | 21.120 |
| 3104 | 5% all-class current CE, old-only replay | 54.800 | 75.300 | 20.500 | 19.689 | 23.200 |
| 3104 | Unchanged current-only comparison | 80.200 | 80.200 | 0.000 | 0.000 | 8.020 |
| 3105 | Original ER, old-only replay | 67.200 | 71.000 | 3.800 | 15.867 | 21.000 |
| 3105 | Restricted incoming, old-only replay | 0.600 | 71.000 | 70.400 | 26.422 | 23.840 |
| 3105 | Full incoming, mixed replay | 71.000 | 74.000 | 3.000 | 16.144 | 21.630 |
| 3105 | Restricted incoming, mixed replay | 66.300 | 73.000 | 6.700 | 18.633 | 23.400 |
| 3105 | 5% all-class current CE, old-only replay | 51.900 | 71.200 | 19.300 | 22.422 | 25.370 |
| 3105 | Unchanged current-only comparison | 71.600 | 71.600 | 0.000 | 0.000 | 7.160 |

## Execution and reproducibility

The sequential mixed-policy job took 25.71 minutes and the 5% job took 13.94 minutes, each within its 55-minute cap. The train-only witness preceded the new matrix, and all 15 endpoints were sealed before test loading. No coefficient, threshold, seed or training-budget search was performed.

The copied archived engine is unchanged. Each model executes 5,530 optimizer updates, with 350,000 scheduled-current and 350,000 replay presentations. Mixed replay contains 175,000 retained old and 175,000 revisited current presentations; old-only replay contains 350,000 old presentations. Both mixed conditions have identical replay indices and augmentation keys. The mixed replay policy changes old and current exposure; these policies are not claimed to have equal old exposure. The original current batch receives its incoming loss, while all replay examples receive all100 CE. The 5% condition uses 0.5 × (0.95 current10 CE + 0.05 all100 current CE + all100 replay CE).

The original three endpoints per seed are reused byte-for-byte. This evidence directory contains 30 sample-level NPZs with official test row identities, mapped/original labels, class order, both predictions, integer counts and NumPy-only recomputation. `counts.csv` and `statistics.json` are derived entirely from those predictions. The scientific protocol excerpt is identified in `protocol.md`. Actual execution sources, completion records and all 1,050 epoch journals are supplied in `execution/`. Full logits and models remain in the authors' working directory.

The hypotheses are conditional directional/sufficiency checks on these fixed histories. They do not determine a minimum sufficient coefficient or a universally optimal memory policy.
