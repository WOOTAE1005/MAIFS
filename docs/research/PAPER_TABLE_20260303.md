# Path A Case3 Multi-Dataset Results (Paper Table)

Updated: 2026-03-03

## Experimental setup

- Datasets: 4 compositions (DS-A/DS-B/DS-C/DS-D)
- Per composition: 300 samples per class (total 900)
- Repeats: split seed 300..309 (`n=10` per composition)
- Metrics: Macro-F1 (mean±std)
- p-value: two-sided exact sign test
- COBRA-only: val-best selection (`algorithm in {rot, drwa, avga, auto}` x `trust profile in {static, metrics_derived}`)

Source JSON:
- `experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260303.json`

## Table 1. 3-case comparison (Macro-F1, mean±std)

| Dataset | COBRA only | DAAC only | COBRA+DAAC | p (DAAC vs COBRA) | p (Fusion vs DAAC) |
|---|---:|---:|---:|---:|---:|
| DS-A (Au/Tp/BigGAN-ai) | 0.277±0.016 | 0.835±0.027 | 0.801±0.028 | 0.00195 | 0.00195 |
| DS-B (Nature/Tp/BigGAN-ai) | 0.274±0.017 | 0.867±0.019 | 0.788±0.041 | 0.00195 | 0.00195 |
| DS-C (Au/IMD/BigGAN-ai) | 0.303±0.019 | 0.898±0.022 | 0.842±0.022 | 0.00195 | 0.00195 |
| DS-D (Nature/IMD/BigGAN-ai) | 0.300±0.021 | 0.863±0.014 | 0.822±0.016 | 0.00195 | 0.00195 |
| **Pooled (40 runs)** | **0.288±0.022** | **0.866±0.031** | **0.813±0.035** | **1.82e-12** | **1.82e-12** |

## Table 2. Best Phase2 vs Best Phase1 (F1 diff, mean±std)

| Dataset | Delta(Phase2-Phase1) | p-value |
|---|---:|---:|
| DS-A | -0.002±0.014 | 1.000 |
| DS-B | +0.002±0.010 | 0.754 |
| DS-C | +0.001±0.015 | 0.508 |
| DS-D | +0.004±0.015 | 0.754 |
| **Pooled (40 runs)** | **+0.001±0.014** | **0.749** |

## Notes

- In all four compositions, `DAAC only > COBRA only` was consistent across all 10 splits.
- In all four compositions, fixed 0.5 fusion (`COBRA+DAAC`) underperformed `DAAC only`.
- The pooled sign-test p-values in Table 1 are for run-level deltas across all 40 runs.
