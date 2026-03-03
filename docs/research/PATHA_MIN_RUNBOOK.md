# Path A Minimal Runbook

Updated: 2026-03-03

## 1. 목적

Path A(실데이터 collector) 실험을 최소 커맨드로 재현한다.

- 기준 config: `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml`
- 결과 저장 루트: `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/`

## 2. 사전 준비

```bash
cd /data/jj812_files/MAIFS
source .venv-qwen/bin/activate
python main.py version
```

## 3. 최소 실행 절차

### Step A) 단일 실행 (artifact/jsonl 생성)

```bash
python experiments/run_phase2_patha.py \
  experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml
```

주요 산출물:
- `phase2_patha_results_*.json`
- `patha_agent_outputs_*.jsonl`
- `phase2_patha_results_*.json` 내부 `three_case_study`
  (`cobra_only`, `daac_only`, `cobra_plus_daac`)

### Step B) 멀티시드 요약 + active gate 자동 평가

```bash
python experiments/run_phase2_patha_multiseed.py \
  experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --summary-out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_20260303.json
```

주요 산출물:
- `summary_10seeds_42_51_20260303.json`
- `summary_10seeds_42_51_20260303_gate_<active_profile>.json`
- `summary_10seeds_42_51_20260303.json` 내부 `aggregate_case3`

### Step C) 고정 데이터셋(k-fold) 재현성 점검

Step A에서 생성된 JSONL 경로를 `<JSONL_PATH>`에 넣어 실행:

```bash
python experiments/run_phase2_patha_repeated.py \
  experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml \
  --precollected-jsonl <JSONL_PATH> \
  --split-strategy kfold \
  --k-folds 5 \
  --kfold-split-seeds 300,301,302,303,304 \
  --summary-out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/fixed_kfold_summary_25runs_300_304_20260303.json
```

주요 산출물:
- `fixed_kfold_summary_25runs_300_304_20260303.json`
- `fixed_kfold_summary_25runs_300_304_20260303_gate_<active_profile>.json`

## 4. Gate 프로파일 일괄 평가 (선택)

```bash
python experiments/evaluate_phase2_gate_profiles.py \
  experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_20260303.json \
  --config experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml \
  --profiles auto \
  --out experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/gate_profiles_10seeds_auto_20260303.json
```

## 5. 최소 확인 항목

요약 JSON에서 아래 값만 먼저 확인:

- `aggregate.f1_diff_mean`
- `aggregate.sign_test_pvalue`
- `aggregate.pooled_mcnemar_pvalue`
- `aggregate.negative_rate`
- gate report의 `report.gate_pass`

## 6. 다중 데이터셋(4조합) 확장 평가

아래 config는 동일 실험틀(300/class, split-seed 300~309)을 다른 데이터 조합으로 실행한다.

- `experiments/configs/tmp_phase2_patha_case3_scale300_dsA.yaml`
- `experiments/configs/tmp_phase2_patha_case3_scale300_dsB.yaml`
- `experiments/configs/tmp_phase2_patha_case3_scale300_dsC.yaml`
- `experiments/configs/tmp_phase2_patha_case3_scale300_dsD.yaml`

조합별 반복 요약:
- `experiments/results/phase2_patha_case3_scale300/repeated_split_summary_10runs_300_309_20260303.json`
- `experiments/results/phase2_patha_case3_scale300_dsB/repeated_split_summary_10runs_300_309_20260303.json`
- `experiments/results/phase2_patha_case3_scale300_dsC/repeated_split_summary_10runs_300_309_20260303.json`
- `experiments/results/phase2_patha_case3_scale300_dsD/repeated_split_summary_10runs_300_309_20260303.json`

통합 비교 산출물:
- `experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260303.json`

논문 표 버전:
- `docs/research/PAPER_TABLE_20260303.md`
