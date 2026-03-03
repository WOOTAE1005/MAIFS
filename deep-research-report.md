# MAIFS (jj8127/MAIFS) — feat/catnet-integration 심층 코드·문서·실험 근거 감사 보고서

**정보 필요(Info Needs)**  
- 런타임(실서비스/CLI)에서 “실제로 적용되는” 신뢰도(trust), 임계값(threshold), 토론(debate) 트리거 조건의 SSOT와 적용 경로  
- DAAC Phase 2의 Path A/B가 각각 어떤 데이터/시드/분할 프로토콜로 재현되며, 결과 JSON이 그 조건을 완전하게 보존하는지  
- CAT-Net(Compression/Frequency 슬롯) 통합이 fallback/UNCERTAIN 정책과 어떻게 결합되어 메타(feature/route) 품질에 어떤 영향을 주는지  
- 실험 게이트(gate profile) 정책이 config/YAML/결과 JSON 사이에서 “동일하게” 해석·적용되는지(특히 active_gate_profile)  
- 테스트가 어떤 실패를 막고 있고(coverage), 어떤 리스크(운영/재현성/정확도)를 아직 방치하는지  

---

## A. Executive Summary (핵심 5줄)

1) GitHub 커넥터로 브랜치 상태(커밋 고정 URL 기준) 코드를 직접 확인했고, 런타임 신뢰도/합의 로직에서 **trust가 중복 적용될 가능성**이 높아 정확도·운영 안정성 리스크가 P0로 확인됐다(예: `src/agents/specialist_agents.py`·`src/maifs.py`·`configs/trust.py`, 라인 **검증 필요**).  
2) `configs/trust.py`가 “trust SSOT”를 선언하지만, 에이전트 내부 `_trust_score`(기본 0.8)와 `configs/settings.py`의 레거시 trust(`semantic` 포함)가 공존해 **설정 SSOT가 분열**되어 있으며, 실험/런타임 간 일관성 저하가 구조적으로 발생한다(`configs/trust.py`, `src/agents/base_agent.py`, `configs/settings.py`, 라인 **검증 필요**).  
3) Path A(실데이터) 파이프라인은 시드 분리(split seed decoupling)·JSONL 재사용(precollected) 기능이 구현돼 있으나, **결과 JSON에 내장된 config 스냅샷과 현재 YAML이 불일치할 수 있는 정황**이 있어 “같은 config로 재현”이 흔들릴 수 있다(`experiments/run_phase2_patha.py`, `experiments/run_phase2_patha_multiseed.py`, `experiments/results/..._gate_*.json`, 라인 **검증 필요**).  
4) CAT-Net 도구는 fallback 시 UNCERTAIN cap을 강제하는 등 graceful degradation은 강하지만, `tool_thresholds.json`의 compression 임계값 설계가 판정 로직과 맞물려 **UNCERTAIN이 사실상 사라질 수 있는 구성**이 존재한다(`src/tools/catnet_tool.py`, `configs/tool_thresholds.json`, 라인 **검증 필요**).  
5) 현재 상태에서 “즉시 main 머지 가능” 판정은 **No**이며, 최소 조건은 (i) trust SSOT 단일화 + 중복 가중 제거, (ii) gate/config 스냅샷 재현성 보강(프로토콜 고정), (iii) debate 트리거/비용 제어, (iv) CAT-Net threshold/UNCERTAIN 정책 정합성 검증이다(근거는 아래 P0 참조).  

**사용한 커넥터(Enabled Connectors)**: `github`  
**GitHub에서 실제로 확인(페치)한 주요 파일 목록(발췌)**:  
- 거버넌스/문서: `AGENTS.md`, `CLAUDE.md`, `PROGRESS_REPORT.md`, `configs/settings.py`, `configs/trust.py`  
- 런타임 코어: `main.py`, `src/maifs.py`, `src/consensus/cobra.py`, `src/debate/debate_chamber.py`, `src/debate/protocols.py`, `src/agents/base_agent.py`, `src/agents/specialist_agents.py`, `src/agents/manager_agent.py`  
- 도구/임계값: `src/tools/catnet_tool.py`, `configs/tool_thresholds.json`  
- DAAC 실험(발췌): `experiments/run_phase2.py`, `experiments/run_phase2_patha.py`, `experiments/run_phase2_patha_multiseed.py`, `experiments/configs/phase2_patha_scale120.yaml`, `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml`  
- DAAC 결과(발췌): `experiments/results/phase2_patha_scale120/..._10seeds_42_51_20260216.json`, `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json`, `.../summary_10seeds_42_51_statsv2_20260216_gate_pooled_relaxed.json`  
- 테스트: `tests/test_meta_collector.py`  

---

## B. Priority Findings (P0/P1/P2)

아래 모든 항목은 **문제 정의 → 근거(파일:라인) → 영향 → 개선안(구현 단위) → 검증 방법(커맨드 포함) → 부작용/롤백** 순서로 기술한다. (라인 번호는 현재 환경에서 자동 산출이 어려워 **“검증 필요”**로 표기한다는 전제에 따름)

### 우선순위 요약 테이블

| Priority | 주제 | 한줄 결론 | 근거(예시) |
|---|---|---|---|
| P0 | Trust 중복 적용(런타임) | agent confidence에 이미 trust가 곱해지고, COBRA에서 다시 trust를 곱할 수 있어 신뢰도 왜곡 가능성 큼 | `src/agents/specialist_agents.py`·`src/agents/base_agent.py`·`src/maifs.py`·`configs/trust.py` (라인 검증 필요) |
| P0 | Gate/Config 재현성(실험) | 결과 JSON·gate report의 config 스냅샷이 “현재 YAML”과 다를 수 있어 동일 프로토콜 A/B가 흔들릴 위험 | `experiments/run_phase2_patha.py`·`experiments/run_phase2_patha_multiseed.py`·`experiments/results/..._gate_*.json` (라인 검증 필요) |
| P0 | Debate 트리거 과도 | verdict가 2개 이상이면 즉시 토론 True → 비용/지연 폭증 가능 | `src/debate/debate_chamber.py` (라인 검증 필요) |
| P1 | CAT-Net threshold 정합성 | compression 임계값이 판정 로직상 “UNCERTAIN 구간 제거”를 유발할 수 있음 | `src/tools/catnet_tool.py`, `configs/tool_thresholds.json` (라인 검증 필요) |
| P1 | SSOT 분열(settings vs trust) | `configs/trust.py` SSOT 선언과 `configs/settings.py` 레거시 trust(semantic 포함) 공존 | `configs/trust.py`, `configs/settings.py` (라인 검증 필요) |
| P2 | Feature profile 명명/차원 | `enhanced36`이 실제로는 37-dim(주석에 명시) → 실험 비교/리포팅 혼선 | `src/meta/collector.py` (라인 검증 필요) |

---

### P0 이슈

#### P0-1. 런타임에서 trust(신뢰도)가 **중복 적용**될 수 있음

**문제 정의**  
에이전트가 반환하는 `AgentResponse.confidence`에 이미 `self.trust_score`(기본 0.8)가 곱해지며, 이후 MAIFS의 COBRA 합의 단계에서 `trust_scores`를 다시 적용한다면(또는 ManagerAgent 경로 사용 시) trust가 2회 반영되어 최종 판정/신뢰도가 왜곡될 수 있다. 이는 “trust score SSOT” 목표와도 충돌한다. (`configs/trust.py`의 SSOT 선언과 상충)

**근거(파일:라인)**  
- 에이전트 레벨 confidence 스케일링: `confidence=tool_result.confidence * self.trust_score` (`src/agents/specialist_agents.py`, `FrequencyAgent.analyze` / `NoiseAgent.analyze` / `FatFormerAgent.analyze` / `SpatialAgent.analyze`, 라인 **검증 필요**)  
- `BaseAgent`의 기본 trust: `_trust_score = 0.8` (`src/agents/base_agent.py`, `BaseAgent.__init__`, 라인 **검증 필요**)  
- MAIFS에서 별도 trust_scores를 COBRA에 전달: `self.trust_scores = resolve_trust()` 후 `consensus_engine.aggregate(agent_responses, self.trust_scores, ...)` (`src/maifs.py`, `MAIFS.__init__` 및 `MAIFS.analyze`, 라인 **검증 필요**)  
- ManagerAgent도 동일 패턴(응답 confidence에 trust를 곱한 다음, 합의에서 또 trust를 사용): `weighted_conf = response.confidence * trust` (`src/agents/manager_agent.py`, `_compute_consensus`, 라인 **검증 필요**)  
- trust SSOT 선언: `configs/trust.py`는 VALID_AGENT_KEYS 기준 trust를 resolve (`configs/trust.py`, `resolve_trust`, 라인 **검증 필요**)

**영향(정확도/재현성/운영 리스크)**  
- **정확도**: 에이전트 confidence 분포가 인위적으로 축소/왜곡되어, 특정 에이전트가 과소평가되거나(특히 0.8 고정 곱) trust_scores의 상대적 효과가 비선형적으로 바뀐다. 이는 합의 알고리즘(DRWA/RoT/AVGA) 입력 분포 자체를 바꿔 판정이 달라질 수 있다(`src/consensus/cobra.py`, 라인 **검증 필요**).  
- **운영 리스크**: trust 튜닝 시 “한 군데만 바꿨는데 영향이 과대/과소”로 나타나 디버깅 난이도가 커진다(SSOT 위반).  
- **재현성**: 실험(Path A/B)에서는 `cobra.trust_scores`를 별도 주입하는데(`experiments/run_phase2_patha.py`, `COBRABaseline`, 라인 **검증 필요**), 런타임과 의미가 달라져 “실험에서 좋았던 설정이 런타임에서 다르게” 작동할 수 있다.

**개선안(구현 단위)**  
- (권장) **정의 단일화**: “trust는 합의 단계에서만 적용”으로 고정. 즉, specialist agent에서 `confidence=tool_result.confidence`로 반환하고, COBRA에서 trust를 반영(`src/agents/specialist_agents.py` 수정).  
- 대안: agent의 `trust_score`를 런타임 초기화 시 `configs/trust.resolve_trust()` 값으로 세팅하고, COBRA에서는 `trust_scores`를 1.0으로 고정(이 경우 “trust 적용 위치”가 agent 레벨로 이동). 단, 이 방식은 합의 알고리즘의 의도(신뢰를 합의에서 다루는 것)와 맞지 않을 수 있다.  
- `ManagerAgent`와 `MAIFS`의 역할 중복이 존재하므로(아래 P0-3), 런타임 진입점을 하나로 고정한 뒤 trust 적용 흐름도 거기에만 남기도록 정리한다.

**검증 방법(실행 커맨드 포함)**  
- (단위) trust 중복 제거 전/후로 agent confidence가 어떻게 변하는지 확인  
  - `pytest tests/ -q` (필수)  
  - 추가 최소 테스트(새로 추가 권장): “agent confidence는 tool confidence와 동일해야 한다”를 검증하는 테스트를 `tests/test_runtime_trust_flow.py`로 추가  
- (런타임 스모크)  
  - `python main.py analyze /path/to/image.jpg --algorithm drwa --device cpu --no-debate` (`main.py`, 라인 **검증 필요**)  
  - 동일 입력에서 최종 `confidence`와 `consensus.agent_weights`의 변화 비교(`src/maifs.py`, `ConsensusResult.to_dict`, 라인 **검증 필요**)

**부작용/롤백 전략**  
- 부작용: 기존 결과에서 confidence 스케일이 달라져 UI/리포트의 “%”가 변동.  
- 롤백: feature flag로 `MAIFS_TRUST_APPLICATION_MODE={agent|consensus}`를 두고(추가 구현), 기본값을 기존 동작으로 유지한 뒤 점진 전환.

---

#### P0-2. Path A 실험 재현성: “현재 YAML”과 “결과 JSON이 내장한 config”가 달라질 수 있음

**문제 정의**  
Path A 결과 파일(특히 gate report)은 실행 시점 config를 내장하지만, repo에 커밋된 YAML이 이후 변경되면 “동일 config 파일 경로”를 재실행해도 동일 조건이 보장되지 않는다. 또한 split.seed decoupling은 구현돼 있으나, 결과 스냅샷에서 seed 정보가 누락되거나(혹은 legacy coupling이 발생) 사람이 “동일 프로토콜 A/B”로 착각할 위험이 있다.

**근거(파일:라인)**  
- split seed decoupling 설계 및 legacy fallback 경고: `split.seed`가 없으면 `collector.seed`로 동조하고 경고를 남김 (`experiments/run_phase2_patha.py`, `run_phase2_patha`, “Seed decoupling 마이그레이션” 블록, 라인 **검증 필요**)  
- multi-seed에서 split/router seed를 루프 seed와 독립 고정: `cfg["split"]["seed"]=base_split_seed`, `cfg["router"]["model"]["random_state"]=base_router_seed` (`experiments/run_phase2_patha_multiseed.py`, `main`, 라인 **검증 필요**)  
- 결과(후보) summary: `experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json`에 `config`는 “파일 경로 문자열”로만 기록(내장 스냅샷이 아니라 경로 참조) (해당 JSON, 라인 **검증 필요**)  
- gate report는 `config`를 통째로 내장: `summary_..._gate_pooled_relaxed.json`가 `config` 블록을 포함(해당 JSON, 라인 **검증 필요**)  
- 동일 결과 파일에서 protocol/gate, split seed 등의 필드가 “현재 YAML”과 다를 여지가 생기는 구조(스냅샷/경로 혼용).

**영향(정확도/재현성/운영 리스크)**  
- **재현성**: “A/B 동일 프로토콜”이 깨져도(예: split.seed 누락으로 coupled) 겉보기엔 같은 config 파일을 쓴 것으로 보일 수 있어, 성능 개선 주장(ΔF1)이 반박 가능하게 검증되지 않는다.  
- **운영 리스크**: gate 기준(예: active_gate_profile)이 문서/설정/결과 중 어디가 진짜인지 혼란 → 머지 의사결정이 흔들림.  
- **정확도**: split이 바뀌면 ΔF1의 분산이 커지고, 통계적 유의성/게이트 통과 여부가 임의로 바뀔 수 있다(실제 seed drift 분석 파일들이 존재하는 정황은 이를 뒷받침).  

**개선안(구현 단위)**  
- **config 스냅샷 SSOT를 “결과 JSON의 embedded config”로 확정**하고, summary에도 `config_snapshot`을 포함하도록 변경  
  - `experiments/run_phase2_patha.py`: `result_path`에 저장되는 결과에는 이미 `seed_meta`, `split` meta 등이 포함됨(현 구조 유지).  
  - `experiments/run_phase2_patha_multiseed.py`: summary JSON에 `base_cfg` 전체를 넣거나, 최소한 `seed_meta`, `split_meta`, `router_random_state`, `collector.seed`, `split.seed`, `datasets.classes`를 명시적으로 상단에 복제 저장.  
- “실행 당시 git commit sha”를 결과에 기록(예: `git rev-parse HEAD` 값을 results JSON에 저장)하여 config drift를 즉시 감지. (구현 위치: `experiments/run_phase2_patha.py` 결과 dict 작성부, 라인 **검증 필요**)  
- gate report 파일명에 “active_gate_profile”만이 아니라 “split.seed / router.random_state / collector.seed / data_source(live|precollected)”를 포함해 사람이 즉시 프로토콜 차이를 인지.

**검증 방법(실행 커맨드 포함)**  
- (필수) `pytest tests/ -q`  
- (동일 프로토콜 A/B 강제)  
  1) 1회 “live collection”로 JSONL을 생성  
     - `.venv-qwen/bin/python experiments/run_phase2_patha.py experiments/configs/phase2_patha_scale120.yaml`  
  2) 생성된 `agent_outputs_jsonl` 경로를 config에 주입해 **완전 동일 데이터**로 baseline/candidate를 각각 실행  
     - (예시: 임시 config 생성)  
       ```bash
       python - <<'PY'
       import yaml, copy
       from pathlib import Path

       base = Path("experiments/configs/phase2_patha_scale120.yaml")
       cand = Path("experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml")

       # 1) 여기 값을 방금 생성된 JSONL로 교체해야 함
       precollected = "experiments/results/phase2_patha_scale120/patha_agent_outputs_YYYYMMDD_HHMMSS.jsonl"

       for src, out in [(base, "tmp_base_precollected.yaml"), (cand, "tmp_cand_precollected.yaml")]:
           cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
           cfg.setdefault("collector", {})["precollected_jsonl"] = precollected
           # split/router seed 고정(동일 프로토콜)
           cfg.setdefault("split", {})["seed"] = 300
           cfg.setdefault("router", {}).setdefault("model", {})["random_state"] = 42
           Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
           print("wrote", out)
       PY
       ```
     - `.venv-qwen/bin/python experiments/run_phase2_patha.py tmp_base_precollected.yaml`  
     - `.venv-qwen/bin/python experiments/run_phase2_patha.py tmp_cand_precollected.yaml`  
- (결과 추출: f1_diff / best model / gate)  
  - 후보 seed10 요약:  
    - `jq '.aggregate.f1_diff_mean, .aggregate.phase2_best_f1_mean, .aggregate.sign_test_pvalue' experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json`  
  - 베이스라인 seed10 요약:  
    - `jq '.aggregate.f1_diff_mean, .aggregate.phase2_best_f1_mean' experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json`  
  - gate 결과:  
    - `jq '.report.gate_pass, .report.candidate.aggregate.f1_diff_mean, .report.candidate.aggregate.pooled_mcnemar_pvalue' experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216_gate_pooled_relaxed.json`

**부작용/롤백 전략**  
- 부작용: 결과 JSON 크기 증가(스냅샷 저장).  
- 롤백: `--compact-config-snapshot` 옵션으로 최소 필드만 저장하는 모드 제공.

---

#### P0-3. 토론(debate) 트리거가 과도해 런타임 비용/지연을 폭발시킬 수 있음

**문제 정의**  
현재 `DebateChamber.should_debate()`는 “판정이 2개 이상으로 나뉘면” 즉시 True를 반환한다. 4개 에이전트 구조상 비동의가 흔한 케이스에서는 거의 항상 토론이 발생할 수 있으며(특히 모델/threshold가 보수적이면 UNCERTAIN 분산↑), 이는 지연/비용/불안정성을 유발한다.

**근거(파일:라인)**  
- 판정 다양성만으로 즉시 토론: `if len(verdicts) >= 2: return True` (`src/debate/debate_chamber.py`, `should_debate`, 라인 **검증 필요**)  
- `MAIFS.analyze()`에서 enable_debate True이고 should_debate True면 즉시 `conduct_debate()` 호출 (`src/maifs.py`, `MAIFS.analyze`, 라인 **검증 필요**)  
- Debating 프로토콜 기본 max_rounds=3 (`src/debate/debate_chamber.py`, `__init__`, 라인 **검증 필요**; `src/debate/protocols.py`, `AsynchronousDebate`, 라인 **검증 필요**)

**영향(정확도/재현성/운영 리스크)**  
- **운영 리스크**: latency 증가(최대 3 라운드), 로그/리포트 복잡도 상승, LLM 연동 시 비용 폭증 가능.  
- **정확도**: 토론이 confidence를 조정하는 로직이 존재하며(`_update_responses_from_debate`, 라인 **검증 필요**), 이 조정이 어떤 데이터에서도 일관된 개선을 보장하지 않는다.  
- **재현성**: 토론은 비결정적 요인이 많아(LLM, 툴 에러, floating differences) 실험과 런타임 비교가 어려워진다.

**개선안(구현 단위)**  
- 토론 트리거를 “단순 불일치”가 아니라 **정량화된 disagreement score**로 통일  
  - 예: COBRA 결과의 `disagreement_level`(엔트로피 기반, `src/consensus/cobra.py`의 RoT는 엔트로피로 계산) 또는 agent verdict distribution 기반 지표를 사용.  
- `DebateChamber.should_debate()`에서 `len(verdicts) >= 2`를 즉시 True로 두지 말고, 예를 들어  
  - `len(verdicts) >= 3` 또는  
  - `max(conf) - min(conf) > threshold`와 같은 조건을 조합하거나,  
  - `uncertain` 포함 여부/비율을 반영해 “불확실성이 큰데 합의가 약한 상황”에서만 토론하도록 제한.  

**검증 방법(실행 커맨드 포함)**  
- (필수) `pytest tests/ -q`  
- (런타임 트레이스) 동일 이미지 입력에서 토론 발생 여부/라운드 수가 논리적으로 감소하는지 확인  
  - `python main.py analyze /path/to/image.jpg --device cpu --algorithm drwa`  
  - `python main.py analyze /path/to/image.jpg --device cpu --algorithm drwa --no-debate`  
- “토론 트리거 단위 테스트” 추가 권장: `should_debate()`가 단순 2-way 불일치에서 False가 되도록 기대값을 명시.

**부작용/롤백 전략**  
- 부작용: 토론이 줄어들어 해석(설명) 풍부함이 감소할 수 있음.  
- 롤백: config로 `debate.trigger_mode={legacy|score}` 토글 제공.

---

### P1 이슈

#### P1-1. CAT-Net compression 임계값과 판정 로직이 UNCERTAIN 설계를 약화시킬 수 있음

**문제 정의**  
CAT-Net 도구는 `manipulation_ratio`를 기반으로 AUTHENTIC/MANIPULATED/UNCERTAIN을 분기한다. 그런데 `configs/tool_thresholds.json`의 `compression.authentic_ratio_threshold`와 `compression.manipulated_ratio_threshold`가 동일(0.0048)인 경우, 코드 구조상 “중간(UNCERTAIN) 구간”이 공집합이 되어 catnet backend에서 UNCERTAIN이 사실상 발생하지 않는다. (fallback/에러 시 UNCERTAIN은 가능)

**근거(파일:라인)**  
- 임계값 로딩 및 적용: `comp_cfg = thresholds["compression"]` 후 `authentic_ratio_threshold`, `manipulated_ratio_threshold` 세팅 (`src/tools/catnet_tool.py`, `CATNetAnalysisTool.__init__`, 라인 **검증 필요**)  
- 판정 로직:  
  - `< authentic_ratio_threshold => AUTHENTIC`  
  - `>= manipulated_ratio_threshold => MANIPULATED`  
  - else UNCERTAIN  
  (`src/tools/catnet_tool.py`, `analyze`, 라인 **검증 필요**)  
- 실제 설정값: `compression.authentic_ratio_threshold = 0.0048`, `compression.manipulated_ratio_threshold = 0.0048` (`configs/tool_thresholds.json`, `compression`, 라인 **검증 필요**)

**영향(정확도/재현성/운영 리스크)**  
- **정확도**: 경계 사례가 MANIPULATED로 과도하게 쏠리면(불확실성의 표현 상실) 다운스트림 합의/메타(feature)에서 오류 신호가 왜곡될 수 있다.  
- **성능 병목/분포**: per-tool verdict 분포에서 UNCERTAIN이 줄어드는 대신 false positive가 늘 가능성이 있어, 트리아지/운영 정책(UNCERTAIN 기반 추가 검토)을 약화시킨다.  
- **재현성**: threshold를 바꿨을 때 UNCERTAIN 분포가 민감하게 반응하지 않으면, 튜닝 효율이 떨어진다.

**개선안(구현 단위)**  
- 임계값 정책을 명시적으로 3-way로 유지하려면 `authentic_ratio_threshold < manipulated_ratio_threshold`를 강제 검증(assert)하고, 같으면 경고/에러.  
- 의도적으로 2-way 분류를 원한다면, 코드에서 else(UNCERTAIN)를 제거하고 문서(AGENTS.md/CLAUDE.md) 및 메타 파이프라인이 이를 전제로 동작하게 정리.  
- `configs/tool_thresholds.json`에 “calibration 기준 데이터/프로토콜”을 더 강하게 기록(현재는 calibrated_at, max_samples만 있음)해 운영 추적성 강화.

**검증 방법(실행 커맨드 포함)**  
- CAT-Net 단독 경계값 테스트(추가 권장):  
  - `pytest tests/ -q`  
  - 새 테스트에서 `authentic_ratio_threshold < manipulated_ratio_threshold`를 검증하거나, 같을 때 UNCERTAIN이 발생하지 않는다는 것을 명시적으로 기대하도록 작성.  
- verdict 분포/UNCERTAIN 사용 여부 추출(아래 F 절 commands 참조).

**부작용/롤백 전략**  
- threshold 변경은 단기적으로 지표(F1/precision/recall) 변동이 큼 → 기존 `tool_thresholds.json`을 백업하고, PR 단위에서 스냅샷 보존.

---

#### P1-2. Trust SSOT는 선언되었으나 settings.py의 레거시 trust/키가 공존

**문제 정의**  
`configs/trust.py`가 trust SSOT를 선언하면서 `_DEPRECATED_KEYS={"semantic"}`를 언급한다. 실제로 `configs/settings.py`의 `COBRAConfig.initial_trust`에 `semantic`이 포함되어 있다. 런타임이 settings의 initial_trust를 쓰지 않더라도, 설정/문서/테스트 코드에서 레거시 경로가 남아 있는 것은 유지보수·혼선을 유발한다.

**근거(파일:라인)**  
- SSOT 및 deprecated 키 정의: `VALID_AGENT_KEYS={"frequency","noise","fatformer","spatial"}`, `_DEPRECATED_KEYS={"semantic"}` (`configs/trust.py`, 라인 **검증 필요**)  
- settings에 semantic trust 존재: `initial_trust` dict에 `"semantic": 0.75` (`configs/settings.py`, `COBRAConfig`, 라인 **검증 필요**)  
- trust를 실제로 resolve하는 런타임 경로: `src/maifs.py`에서 `resolve_trust()` 호출 (`src/maifs.py`, 라인 **검증 필요**)

**영향(정확도/재현성/운영 리스크)**  
- **운영 리스크**: 누군가 settings의 initial_trust를 수정해도 런타임이 반영하지 않으면 “설정 변경이 반영되지 않는” 장애 양상을 만든다.  
- **재현성**: 실험 config(YAML)에서 trust override를 넣는 경우(`split`/`cobra` 설정), trust SSOT가 어디인지 더 혼란스러워진다.

**개선안(구현 단위)**  
- settings에서 trust dict를 제거하거나, ship 단계에서 “settings.cobra.initial_trust는 사용하지 않음”을 명시하고 `resolve_trust()`로 완전 위임.  
- `configs/settings.py`가 필요하다면, trust는 `configs/trust.py`를 import해서 참조하도록 역참조(SSOT 강제).

**검증 방법(실행 커맨드 포함)**  
- `pytest tests/ -q`  
- (정적 검증) `semantic` 키가 enumerate에 남아 있는지 grep:  
  - `rg -n '"semantic"' configs/ src/ experiments/`

**부작용/롤백 전략**  
- 문서/테스트가 settings를 전제로 하는 경우 깨질 수 있음 → deprecation 단계로 처리(경고 → 삭제).

---

### P2 이슈

#### P2-1. proxy feature profile 명명/차원 불일치가 리포팅 혼선을 유발

**문제 정의**  
프로젝트는 `enhanced36`을 기능 프로파일 이름으로 쓰지만, 실제 구현 주석에 따르면 차원이 37이다(“이름은 36이나 실제 차원은 37”). 실험 비교(특히 feature ablation)에서 “차원/feature set” 혼동이 발생할 수 있다.

**근거(파일:라인)**  
- `enhanced36` 설명 및 실제 차원 언급: `build_proxy_image_features(... feature_set="enhanced36")` 주석에 “실제 차원 37” 명시 (`src/meta/collector.py`, `build_proxy_image_features` docstring, 라인 **검증 필요**)  
- `risk52`, `evidence_2ch`도 위 프로파일을 기반으로 차원을 계산 (`src/meta/collector.py`, 동일, 라인 **검증 필요**)  
- 관련 테스트는 “shape”만 검증(`risk52`는 52 shape, base는 20 shape) (`tests/test_meta_collector.py`, 라인 **검증 필요**)

**영향(정확도/재현성/운영 리스크)**  
- **재현성**: “enhanced36”이라는 이름만으로는 실제 입력 차원을 알기 어려워, 외부 분석/논문 도표에서 오기 쉬움.  
- **운영 리스크**: feature 변경 시 gate policy/튜닝이 꼬일 가능성.

**개선안(구현 단위)**  
- 프로파일명을 `enhanced37`로 변경(가장 명확).  
- 또는 `feature_dim`을 명시적으로 결과 JSON에 기록(이미 `run_phase2_patha.py`가 `proxy_feature_dim`을 저장함: `results["router"]["proxy_feature_dim"]`, 라인 **검증 필요**)하고, 문서는 그 수치를 SSOT로 삼는 방식.

**검증 방법(실행 커맨드 포함)**  
- `pytest tests/ -q`  
- 결과 JSON에서 `proxy_feature_dim`을 확인:  
  - `jq '.router.proxy_feature_dim, .router.feature_set' <phase2_patha_results_*.json>`

**부작용/롤백 전략**  
- 파일명/키 변경은 downstream 스크립트에 영향 → `enhanced36` alias를 일정 기간 유지.

---

## C. Quick Win Top 5 (노력 대비 효과 순)

| Rank | Quick Win | 기대 효과 | 변경 범위(예상) | 근거 |
|---|---|---|---|---|
| 1 | **trust 중복 제거(에이전트 confidence 스케일링 제거)** | 합의 안정화, 튜닝 가능성↑, 결과 해석 용이 | `src/agents/specialist_agents.py`, (필요 시) `src/agents/manager_agent.py`, `src/maifs.py` | `specialist_agents`에서 confidence에 trust 곱, `maifs`에서 trust_scores 재적용 (라인 검증 필요) |
| 2 | debate 트리거를 disagreement score 기반으로 완화 | latency/비용 급감, 운영 안전성↑ | `src/debate/debate_chamber.py`, `src/maifs.py` | verdict 2종 이상이면 무조건 토론 True (라인 검증 필요) |
| 3 | gate/summary에 **config_snapshot + git sha** 저장 | 실험 재현성 급상승, 리뷰 반박 가능성↑ | `experiments/run_phase2_patha.py`, `experiments/run_phase2_patha_multiseed.py` | summary가 config 경로만 저장하는 경우 존재 (라인 검증 필요) |
| 4 | CAT-Net threshold 정합성 체크(UNCERTAIN 구간 정책 명확화) | per-tool 분포 통제, 오탐/불확실성 정책 일치 | `src/tools/catnet_tool.py`, `configs/tool_thresholds.json` | authentic/manip threshold 동일 가능 (라인 검증 필요) |
| 5 | PROGRESS_REPORT/CLAUDE/AGENTS SSOT 문장 정리(“진짜 SSOT는 AGENTS”) | 온보딩 비용↓, 잘못된 수정 방지 | `PROGRESS_REPORT.md`, `CLAUDE.md`, `AGENTS.md` | CLAUDE에서 “SSOT” 주장, AGENTS에서 “SSOT” 선언(라인 검증 필요) |

---

## D. 30/60/90 로드맵

**30일(merge-blocker 제거)**  
- trust 적용 위치를 단일화(권장: 합의 단계만)하고, 관련 단위 테스트 추가: “agent confidence == tool confidence” + “COBRA trust 적용 1회” (`src/agents/*`, `tests/`) (라인 검증 필요).  
- debate 트리거 정책을 변경(불일치 2개 이상 → 조건부)하고, 최소한 `--no-debate` 모드가 완전 결정적으로 동작하도록 스모크를 고정 (`src/debate/debate_chamber.py`, `main.py`, 라인 검증 필요).  
- Path A 결과 산출물에 config_snapshot + git sha + seed_meta를 summary/gate에 **강제 포함** (`experiments/run_phase2_patha*.py`, 라인 검증 필요).  

**60일(성능·재현성 강화)**  
- 동일 데이터(=precollected JSONL) 기반의 “baseline vs candidate(동일 split/router seed)” 표준 A/B 프로토콜을 문서+스크립트로 고정하고, 자동 리포트 생성(`f1_diff/best_model/gate_pass/verdict_distribution/fallback_rate`)까지 일괄 산출 (`experiments/`, 라인 검증 필요).  
- CAT-Net/FatFormer/Mesorch 등 optional dependency 미설치 시 fallback이 발생할 때, **fallback rate를 결과 JSON에 상시 기록**하고 gate 조건에 포함(예: fallback_rate <= X) (`src/tools/catnet_tool.py`, `src/meta/collector.py`, 라인 검증 필요).  

**90일(운영 품질/연구 품질 동시 달성)**  
- Path A의 variance 분해(collector seed vs split seed vs router seed)를 표준화하고, gate profile을 보수형/공격형/손실회피형으로 명확히 분리해 운영 정책에 연결 (`experiments/run_phase2_patha_multiseed.py`, `src/meta/router.py`, 라인 검증 필요).  
- 런타임 MAIFS 경로와 연구(DAAC) 경로 간의 “용어/신뢰도/판정 계약”을 단일 문서로 통합(AGENTS.md 중심)하고, 문서 자동 동기화 체크(간단한 CI) 추가.

---

## E. PR Plan

### 머지 판정
**지금 당장 main 머지 가능? → No.**  
**조건부 Yes**: 아래 “Merge Checklist”의 P0 항목이 모두 충족되고, 최소 1회 “동일 precollected JSONL 기반 A/B”에서 결과가 재현(동일 seed_meta/동일 split)될 때.

### 제안 PR 분해(브랜치/변경 파일/규모/테스트/체크리스트)

#### PR-1: trust SSOT 단일화 및 중복 가중 제거
- 브랜치명: `fix/trust-ssot-single-application`
- 변경 파일(예상):
  - `src/agents/specialist_agents.py` (confidence 스케일링 제거 또는 모드화) (라인 검증 필요)
  - (필요 시) `src/agents/manager_agent.py` (중복 적용 제거/정리) (라인 검증 필요)
  - `tests/` 신규: `test_runtime_trust_flow.py` (신규)  
- 예상 diff 규모: 중간(수십~200 LOC)
- 테스트:
  - `pytest tests/ -q`
- Merge checklist:
  - [ ] trust가 1회만 적용됨을 단위 테스트로 보장  
  - [ ] 런타임 출력(최종 confidence) 분포 변화가 의도된 것임을 AGENTS.md에 기록  

#### PR-2: debate 트리거 정책 개선 및 비용 제어
- 브랜치명: `fix/debate-trigger-policy`
- 변경 파일(예상):
  - `src/debate/debate_chamber.py` (`should_debate` 로직 개선) (라인 검증 필요)
  - (필요 시) `src/maifs.py` (토론 적용 조건/로깅 개선) (라인 검증 필요)
  - `tests/` 신규: `test_debate_trigger.py` (신규)  
- 예상 diff 규모: 소~중
- 테스트:
  - `pytest tests/ -q`

#### PR-3: Path A/B 결과물 재현성 강화(config snapshot + git sha + seed meta 표준화)
- 브랜치명: `fix/experiments-repro-snapshot`
- 변경 파일(예상):
  - `experiments/run_phase2_patha.py` (결과 JSON에 git sha 포함) (라인 검증 필요)
  - `experiments/run_phase2_patha_multiseed.py` (summary에 config_snapshot/seed_meta 상단 고정) (라인 검증 필요)
- 예상 diff 규모: 중
- 테스트:
  - `pytest tests/ -q`
  - 스모크: `.venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120.yaml --seeds 42,43` (환경/데이터 필요)

#### PR-4: CAT-Net threshold 정책 명확화(UNCERTAIN 구간)
- 브랜치명: `fix/catnet-threshold-policy`
- 변경 파일(예상):
  - `src/tools/catnet_tool.py` (threshold 관계 검증 또는 2-way 전환 명시) (라인 검증 필요)
  - `configs/tool_thresholds.json` (정책에 맞는 값으로 조정) (라인 검증 필요)
  - 테스트: `tests/test_catnet_threshold_policy.py` (신규)  
- 예상 diff 규모: 소

---

## F. 즉시 실행 가능한 검증 커맨드 목록

아래는 “지금 당장” 실행 가능한 형태로 정리했다. (데이터셋 경로는 결과 JSON에 이미 등장하는 경로를 기준으로 작성: `datasets/CASIA2_subset/*`, `datasets/GenImage_subset/BigGAN/val/ai` 등은 환경에 존재해야 함 — `experiments/results/..._gate_*.json`의 `config.datasets` 참조, 라인 **검증 필요**)

### 필수: 테스트
```bash
pytest tests/ -q
```

### Path A 단일 실행(A/B 비교 — 동일 프로토콜 강제 버전)
1) **baseline 1회 실행**(live collection → JSONL 생성)
```bash
.venv-qwen/bin/python experiments/run_phase2_patha.py experiments/configs/phase2_patha_scale120.yaml
```

2) 생성된 결과 JSON에서 `agent_outputs_jsonl` 경로를 확인(예: `experiments/results/phase2_patha_scale120/patha_agent_outputs_*.jsonl`)  
(결과 JSON 구조는 `experiments/run_phase2_patha.py`의 `results["artifacts"]["agent_outputs_jsonl"]` 생성부를 따름, 라인 **검증 필요**)

3) 동일 JSONL을 baseline/candidate에 주입한 임시 config 생성 후 각각 1회 실행(동일 데이터, 동일 split/router seed)
```bash
python - <<'PY'
import yaml
from pathlib import Path

precollected = "experiments/results/phase2_patha_scale120/patha_agent_outputs_YYYYMMDD_HHMMSS.jsonl"

pairs = [
  ("experiments/configs/phase2_patha_scale120.yaml", "tmp_base_precollected.yaml"),
  ("experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml", "tmp_cand_precollected.yaml"),
]
for src, out in pairs:
    cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    cfg.setdefault("collector", {})["precollected_jsonl"] = precollected
    cfg.setdefault("split", {})["seed"] = 300
    cfg.setdefault("router", {}).setdefault("model", {})["random_state"] = 42
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("wrote", out)
PY

.venv-qwen/bin/python experiments/run_phase2_patha.py tmp_base_precollected.yaml
.venv-qwen/bin/python experiments/run_phase2_patha.py tmp_cand_precollected.yaml
```

### 결과 JSON에서 f1_diff / best model / gate 결과 추출

#### (A) seed10 요약에서 f1_diff_mean / best model 리스트 / sign-test / pooled McNemar
- 후보(`enhanced36+ridge`) 요약:
```bash
jq '.aggregate.f1_diff_mean,
    .aggregate.phase2_best_f1_mean,
    .aggregate.sign_test_pvalue,
    .aggregate.pooled_mcnemar_pvalue,
    (.runs | map({seed, phase1_best, phase2_best, f1_diff}))' \
  experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216.json
```

- 베이스라인 요약:
```bash
jq '.aggregate.f1_diff_mean,
    .aggregate.phase2_best_f1_mean,
    (.runs | map({seed, phase1_best, phase2_best, f1_diff}))' \
  experiments/results/phase2_patha_scale120/phase2_patha_multiseed_summary_scale120_10seeds_42_51_20260216.json
```

#### (B) gate report에서 gate_pass / 기준 / 후보-베이스라인 비교 요약
```bash
jq '.active_gate_profile,
    .report.gate_pass,
    .report.criteria,
    .report.candidate.aggregate.f1_diff_mean,
    .report.candidate.aggregate.sign_test_pvalue,
    .report.candidate.aggregate.pooled_mcnemar_pvalue,
    .report.baseline_delta_f1_diff_mean' \
  experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/summary_10seeds_42_51_statsv2_20260216_gate_pooled_relaxed.json
```

### 성능 병목 점검: per-tool verdict 분포 / UNCERTAIN 사용 / fallback rate

Path A는 agent output JSONL에 `agent_verdicts`와 `evidence_digest`가 저장되도록 설계되어 있고(`src/meta/collector.py`, `CollectedRecord` 및 `save_jsonl`, 라인 **검증 필요**), 이를 이용해 도구별 분포/폴백을 직접 산출할 수 있다.

#### (A) 도구별 verdict 분포
```bash
python - <<'PY'
import json
import collections

path = "experiments/results/phase2_patha_scale120/patha_agent_outputs_20260216_103004.jsonl"  # 예시(실제 파일로 교체)
agents = ["frequency","noise","fatformer","spatial"]
counts = {a: collections.Counter() for a in agents}
n = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        for a,v in d["agent_verdicts"].items():
            counts[a][v] += 1
        n += 1

print("N =", n)
for a in agents:
    print(a, dict(counts[a]))
PY
```

#### (B) frequency 슬롯(CAT-Net)의 fallback 발동률 및 UNCERTAIN cap 여부
- CAT-Net fallback 시 `evidence.backend == "frequency_fallback"`, `fallback_mode=True`, `fallback_raw_*`를 기록하도록 구현됨(`src/tools/catnet_tool.py`, fallback 분기, 라인 **검증 필요**; `src/meta/collector.py`의 evidence schema에도 관련 키 포함, 라인 **검증 필요**).
```bash
python - <<'PY'
import json
import collections

path = "experiments/results/phase2_patha_scale120/patha_agent_outputs_20260216_103004.jsonl"  # 예시(실제 파일로 교체)
fallback = 0
uncertain = 0
n = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        n += 1
        freq_v = d["agent_verdicts"].get("frequency")
        if freq_v == "uncertain":
            uncertain += 1
        ev = (d.get("evidence_digest") or {}).get("frequency") or {}
        if ev.get("backend") == "frequency_fallback" or ev.get("fallback_mode") is True:
            fallback += 1

print("N =", n)
print("frequency_uncertain_rate =", uncertain / max(1,n))
print("frequency_fallback_rate  =", fallback / max(1,n))
PY
```

### 설정 일관성 점검: trust SSOT 적용 여부(런타임)
- 런타임은 `resolve_trust()`를 사용(`src/maifs.py`, 라인 **검증 필요**)하지만, 에이전트 내부 trust(`BaseAgent._trust_score`)도 존재하므로 중복이 실제로 발생하는지 로그로 확인:
```bash
python - <<'PY'
from src.maifs import MAIFS
m = MAIFS(enable_debate=False, device="cpu")
print("maifs.trust_scores(from configs/trust.py) =", m.trust_scores)
for name, agent in m.agents.items():
    print(name, "agent.trust_score(BaseAgent) =", getattr(agent, "trust_score", None))
PY
```

(위 출력에서 `agent.trust_score`가 1.0이 아닌 값(기본 0.8)이고, `maifs.trust_scores`도 별도로 존재하면 “중복 적용 가능성”이 실증된다. 관련 근거: `src/agents/base_agent.py`·`src/agents/specialist_agents.py`·`src/maifs.py`, 라인 **검증 필요**)

---