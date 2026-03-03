# MAIFS Current Architecture Spec

Updated: 2026-03-03

## 1. Scope and Authority

This document is the code-first architecture SSOT for MAIFS runtime and DAAC experiment flow.

Authoritative code paths:
- `src/maifs.py`
- `src/agents/specialist_agents.py`
- `src/agents/manager_agent.py`
- `src/consensus/cobra.py`
- `src/debate/debate_chamber.py`
- `src/tools/*.py`
- `src/meta/*.py`
- `experiments/run_phase*.py`

Final verdict contract (immutable):
- `authentic`
- `manipulated`
- `ai_generated`
- `uncertain`

## 2. Runtime Layering

1. Tool layer
- Frequency slot: `CATNetAnalysisTool` with `FrequencyAnalysisTool` fallback.
- Noise slot: `NoiseAnalysisTool` (`mvss` or `prnu` backend).
- FatFormer slot: `FatFormerTool`.
- Spatial slot: `SpatialAnalysisTool` (default `mesorch`, optional `trufor`) with safe fallback.

2. Agent layer
- Specialists: `FrequencyAgent`, `NoiseAgent`, `FatFormerAgent`, `SpatialAgent`.
- Input: image ndarray.
- Output: `AgentResponse(verdict, confidence, evidence, reasoning, arguments)`.

3. Consensus/debate layer
- Consensus engine: `COBRAConsensus` (`rot`/`drwa`/`avga`).
- Debate engine: `DebateChamber` with protocol-based rounds.
- Debate is conditional on disagreement threshold.

4. Entry layer
- Canonical runtime entry: `MAIFS.analyze()` in `src/maifs.py`.
- `ManagerAgent` is aligned to the same COBRA+Debate engines for consistency.

## 3. Confidence and Trust Contract

Rule:
- Specialist response confidence is raw tool confidence (`tool_result.confidence`).
- Agent trust score is applied once during consensus aggregation (COBRA/manager consensus stage).
- Debate stage confidence update is trust-neutral (challenge strength only).

This removes double-weighting risk from the old pattern:
- old: `response.confidence = tool_conf * trust` and then COBRA applied trust again.
- current: `response.confidence = tool_conf`; trust weighting only in consensus.
- old(debate): challenger trust scaled confidence delta during debate, then COBRA trust applied again.
- current(debate): debate updates confidence without trust multiplier; trust is applied only at consensus.

## 4. Manager and MAIFS Path Alignment

Alignment policy:
- `MAIFS` and `ManagerAgent` both use `COBRAConsensus` and `DebateChamber`.
- `ManagerAgent` no longer uses independent custom consensus math as primary runtime path.
- LLM report generation in `ManagerAgent.analyze_with_llm()` operates on the same consensus/debate outputs.

## 5. Fallback Contract

Hard requirements:
- Missing checkpoints/dependencies must not crash full analysis.
- Tool should return `UNCERTAIN` with explicit evidence flags when fallback is used.

Examples:
- CAT-Net unavailable: frequency fallback runs, raw fallback verdict stored in evidence, returned verdict capped to uncertain-safe mode.
- FatFormer/Spatial model unavailable: deterministic fallback path with explicit `fallback_mode` evidence.

## 6. DAAC Architecture (Phase 1/2)

Path B (simulation):
- Phase 1: disagreement feature learning (`43-dim`) vs baseline COBRA/majority.
- Phase 2: adaptive routing (`47-dim = 43 + 4 router weights`).

Path A (real-data collector):
- Collect specialist outputs from configured datasets.
- Train/evaluate phase1 vs phase2 with protocolized split.
- Evaluate using gate profiles (strict/sign-driven/loss-averse variants).

Key files:
- `experiments/run_phase1.py`
- `experiments/run_phase2.py`
- `experiments/run_phase2_patha.py`
- `experiments/run_phase2_patha_multiseed.py`
- `experiments/run_phase2_patha_repeated.py`
- `experiments/evaluate_phase2_gate.py`

## 7. Terminology and Deprecation Map

Runtime terminology:
- Use `fatformer` slot name.
- Do not use runtime `watermark` slot terminology.
- `semantic` trust key is deprecated and ignored by trust resolver.

Historical/proposal documents may still include watermark-era terms and must be treated as archival unless explicitly synchronized.

## 8. Recent Validation Snapshot (2026-03-03)

- Debate trust-double-count mitigation:
  - `src/debate/debate_chamber.py` now removes challenger-trust scaling in debate confidence updates.
  - Regression test added: `tests/test_debate.py::test_debate_confidence_update_is_trust_neutral`.
- Path A multi-dataset case3 evaluation:
  - 4 dataset compositions, each 300/class, split-seed 10 repeats.
  - Consolidated output:
    - `experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260303.json`
    - `docs/research/PAPER_TABLE_20260303.md`
