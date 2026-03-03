# MAIFS Operations Risk Priorities

Updated: 2026-03-03

## Resolved (2026-03-03): Trust Double-Weighting in Debate Path

Previous issue:
- Debate stage used challenger trust to scale confidence deltas, and COBRA consensus also applied trust.

Action:
- Removed trust multiplier from debate confidence updates.
- Kept trust weighting exclusively in consensus aggregation.

Code/tests:
- `src/debate/debate_chamber.py`
- `tests/test_debate.py::test_debate_confidence_update_is_trust_neutral`

Residual risk:
- None for runtime trust double-count on debate path under current code.
- Monitor future debate logic changes for reintroduction.

## Priority 1: Fallback Quality Drift

Risk:
- Optional backend dependency/checkpoint 누락 시 fallback 비율이 높아지고 성능이 급락할 수 있음.

Impact:
- High (운영 판정 신뢰도 직접 저하)

Signals:
- `fallback_mode` evidence 비율 증가
- 특정 슬롯의 `UNCERTAIN` 급증

Immediate controls:
- CI/배포 전 `scripts/evaluate_tools.py` smoke 결과를 baseline과 비교
- `requirements-optional-tools.txt` 핀 유지
- 체크포인트 존재 검증을 배포 체크리스트에 포함

## Priority 2: Path A Gate Overfitting or Mis-calibration

Risk:
- 단일 seed block 기준으로 gate 임계값을 고정하면 block drift에서 fail 가능.

Impact:
- High (승인/배포 결정 오류)

Signals:
- block 간 `f1_diff_mean` 부호 전환
- sign-test/pooled McNemar 방향성 불안정

Immediate controls:
- fixed-kfold + repeated-split 동시 기준 유지
- active gate profile 변경 시 독립 block 재검증 필수
- downside 지표(`negative_rate`, `downside_mean`, `cvar_downside`) 병행 모니터링

## Priority 3: Runtime Path Divergence

Risk:
- MAIFS와 Manager 경로가 다른 합의 규칙을 쓰면 디버깅과 결과 해석이 분리됨.

Impact:
- Medium-High

Current mitigation:
- ManagerAgent를 COBRA+Debate 공통 엔진으로 정렬 완료.

Ongoing controls:
- 합의 로직 변경은 `src/consensus/cobra.py` 중심으로만 수행
- 관련 회귀 테스트 추가/유지

## Priority 4: Dependency and Environment Drift

Risk:
- CUDA, torch, opencv, numpy ABI 조합 드리프트로 특정 슬롯만 은밀히 실패.

Impact:
- Medium

Signals:
- 환경별 재현 불일치
- 특정 백엔드만 에러/성능 저하

Immediate controls:
- 환경 스냅샷 기록(핵심 패키지 버전)
- smoke 커맨드 표준화 및 결과 JSON 보관

## Priority 5: Documentation Drift

Risk:
- 역사 문서(제안서/초기 설계)와 현재 코드가 혼재되어 온보딩 오류 발생.

Impact:
- Medium

Immediate controls:
- 현재 문서를 SSOT로 지정:
  - `docs/research/CURRENT_ARCHITECTURE_SPEC.md`
  - `docs/research/PATHA_MIN_RUNBOOK.md`
- README에서 SSOT 링크를 최상단 노출
