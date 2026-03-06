# MAIFS vs SOTA Snapshot (2026-03-04)

## 1) MAIFS 6-dataset internal comparison (3-class macro-F1)

| Dataset | COBRA-only | DAAC-only | COBRA+DAAC |
|---|---:|---:|---:|
| DS-A_CASIA+BigGAN | 0.2771 ± 0.0171 | 0.8353 ± 0.0282 | 0.8012 ± 0.0297 |
| DS-B_ImageNet+CASIA+BigGAN | 0.2738 ± 0.0177 | 0.8672 ± 0.0203 | 0.7882 ± 0.0435 |
| DS-C_CASIA+IMD2020+BigGAN | 0.3027 ± 0.0198 | 0.8976 ± 0.0232 | 0.8423 ± 0.0236 |
| DS-D_ImageNet+IMD2020+BigGAN | 0.2995 ± 0.0224 | 0.8634 ± 0.0151 | 0.8215 ± 0.0165 |
| OpenSDI_subset300 | 0.1772 ± 0.0121 | 0.4094 ± 0.0362 | 0.2207 ± 0.0422 |
| AI-GenBench_proxy300 | 0.3332 ± 0.0225 | 0.7502 ± 0.0199 | 0.6168 ± 0.0248 |

- Overall (60 runs): COBRA=0.2772, DAAC=0.7705, COBRA+DAAC=0.6818

## 2) External leaderboard snapshots (official pages)

### OpenSDI leaderboard (2025-11-03 snapshot)

- Top: GMMDet F1 Avg 94.16 | DIRE 91.62 | NPR 88.66 | UniFD 86.95 | RINE 84.44
- Source: https://opensdi.github.io/

### AI-GenBench leaderboard (last updated 2025-02-17)

- Top: CLOSER Acc 95.31 / AUC 99.16 / EER 1.97; Msm-detection 93.18 / 98.89 / 2.57; NPR 89.77 / 97.08 / 5.42
- Source: https://mi-biolab.github.io/ai-genbench/

## 3) Comparison caveat (important)

- MAIFS results here are **3-class macro-F1 on local subset/proxy protocols**.
- Leaderboard results are **official protocol metrics** (mostly binary detection and different split design).
- Therefore, direct numeric superiority claims are not statistically valid yet.

## 4) Restricted dataset readiness

| Dataset | URL status | Local ready | Note |
|---|---:|---:|---|
| coverage | 200 | False |  |
| columbia | 403 | False |  |
| nist16 | 401 | False |  |
