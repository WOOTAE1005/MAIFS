# COBRA Mismatch Diagnostics

- jsonl: `experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl`
- seeds: `[300, 301, 302, 303, 304, 305, 306, 307, 308, 309]`
- baseline: `DRWA + static trust + default params` = **0.2664 ± 0.0090**

## A. Trust/Algorithm Grid

| Setting | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |
|---|---:|---:|---:|---:|
| auto__metrics_derived_train | 0.2774 ± 0.0178 | 0.0110 ± 0.0139 | 0.0371 | 8/2/0 |
| auto__oracle_test_upper | 0.2774 ± 0.0178 | 0.0110 ± 0.0139 | 0.0371 | 8/2/0 |
| auto__shuffled_static | 0.2765 ± 0.0172 | 0.0101 ± 0.0136 | 0.0840 | 6/4/0 |
| auto__static | 0.2763 ± 0.0171 | 0.0099 ± 0.0135 | 0.0840 | 6/4/0 |
| avga__metrics_derived_train | 0.2775 ± 0.0177 | 0.0111 ± 0.0138 | 0.0371 | 8/2/0 |
| avga__oracle_test_upper | 0.2775 ± 0.0176 | 0.0111 ± 0.0137 | 0.0371 | 8/2/0 |
| avga__shuffled_static | 0.2779 ± 0.0177 | 0.0115 ± 0.0137 | 0.0371 | 8/2/0 |
| avga__static | 0.2780 ± 0.0177 | 0.0116 ± 0.0137 | 0.0371 | 8/2/0 |
| drwa__metrics_derived_train | 0.2656 ± 0.0106 | -0.0008 ± 0.0049 | 0.6523 | 3/6/1 |
| drwa__oracle_test_upper | 0.2649 ± 0.0104 | -0.0015 ± 0.0053 | 0.4316 | 3/7/0 |
| drwa__shuffled_static | 0.2660 ± 0.0093 | -0.0004 ± 0.0010 | 0.2383 | 4/5/1 |
| drwa__static | 0.2664 ± 0.0090 | 0.0000 ± 0.0000 | 1.0000 | 0/0/10 |
| rot__metrics_derived_train | 0.2663 ± 0.0125 | -0.0001 ± 0.0062 | 0.8457 | 5/5/0 |
| rot__oracle_test_upper | 0.2629 ± 0.0102 | -0.0035 ± 0.0055 | 0.1602 | 4/6/0 |
| rot__shuffled_static | 0.2683 ± 0.0125 | 0.0019 ± 0.0062 | 0.4316 | 5/5/0 |
| rot__static | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |

## B. Sensitivity (static trust)

### RoT trust_threshold

| Param | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |
|---|---:|---:|---:|---:|
| rot_threshold=0.500 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |
| rot_threshold=0.600 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |
| rot_threshold=0.700 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |
| rot_threshold=0.800 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |
| rot_threshold=0.900 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |

### DRWA epsilon

| Param | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |
|---|---:|---:|---:|---:|
| drwa_epsilon=0.000 | 0.2664 ± 0.0128 | 0.0000 ± 0.0071 | 1.0000 | 4/6/0 |
| drwa_epsilon=0.100 | 0.2696 ± 0.0128 | 0.0032 ± 0.0053 | 0.0938 | 5/1/4 |
| drwa_epsilon=0.200 | 0.2664 ± 0.0090 | 0.0000 ± 0.0000 | 1.0000 | 0/0/10 |
| drwa_epsilon=0.400 | 0.2637 ± 0.0103 | -0.0027 ± 0.0035 | 0.1250 | 0/4/6 |
| drwa_epsilon=0.800 | 0.2685 ± 0.0104 | 0.0021 ± 0.0056 | 0.2500 | 3/1/6 |

### AVGA temperature

| Param | Macro-F1 | Delta vs baseline | Wilcoxon p | sign(+/-/0) |
|---|---:|---:|---:|---:|
| avga_temperature=0.500 | 0.2723 ± 0.0124 | 0.0059 ± 0.0051 | 0.0156 | 7/0/3 |
| avga_temperature=1.000 | 0.2780 ± 0.0177 | 0.0116 ± 0.0137 | 0.0371 | 8/2/0 |
| avga_temperature=1.500 | 0.2774 ± 0.0178 | 0.0110 ± 0.0139 | 0.0371 | 8/2/0 |
| avga_temperature=2.000 | 0.2774 ± 0.0178 | 0.0110 ± 0.0139 | 0.0371 | 8/2/0 |
| avga_temperature=3.000 | 0.2774 ± 0.0178 | 0.0110 ± 0.0139 | 0.0371 | 8/2/0 |

