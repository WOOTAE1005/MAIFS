"""
Runtime trust flow and manager/maifs alignment regression tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.base_agent import AgentResponse, AgentRole
from src.agents.manager_agent import ManagerAgent
from src.agents.specialist_agents import (
    FatFormerAgent,
    FrequencyAgent,
    NoiseAgent,
    SpatialAgent,
)
from configs.trust import resolve_trust, derive_trust_from_metrics
from src.maifs import MAIFS
from src.tools.base_tool import ToolResult, Verdict


def _mock_tool_result(confidence: float) -> ToolResult:
    return ToolResult(
        tool_name="stub-tool",
        verdict=Verdict.MANIPULATED,
        confidence=confidence,
        evidence={"stub": True},
        explanation="stub",
    )


@pytest.mark.parametrize(
    ("agent_cls", "role"),
    [
        (FrequencyAgent, AgentRole.FREQUENCY),
        (NoiseAgent, AgentRole.NOISE),
        (FatFormerAgent, AgentRole.FATFORMER),
        (SpatialAgent, AgentRole.SPATIAL),
    ],
)
def test_specialist_analyze_returns_raw_tool_confidence(agent_cls, role):
    """
    Specialist confidence must not be pre-multiplied by agent trust.
    """

    class StubAgent:
        pass

    tool_conf = 0.73
    stub = StubAgent()
    stub.name = "stub-agent"
    stub.role = role
    stub.trust_score = 0.01  # should not affect returned confidence
    stub._tool = lambda image: _mock_tool_result(tool_conf)
    stub.generate_reasoning = lambda tool_results, context=None: "stub reasoning"
    stub._extract_arguments = lambda tool_result: ["stub argument"]

    response = agent_cls.analyze(stub, np.zeros((8, 8, 3), dtype=np.uint8))

    assert response.confidence == pytest.approx(tool_conf)
    assert response.verdict == Verdict.MANIPULATED


def test_manager_consensus_pipeline_matches_cobra_aggregate():
    """
    Manager consensus path must be identical to COBRA aggregate when debate is disabled.
    """
    manager = ManagerAgent(use_llm=False, enable_debate=False, consensus_algorithm="drwa")
    responses = {
        "frequency": AgentResponse(
            agent_name="freq",
            role=AgentRole.FREQUENCY,
            verdict=Verdict.MANIPULATED,
            confidence=0.70,
            reasoning="r1",
        ),
        "noise": AgentResponse(
            agent_name="noise",
            role=AgentRole.NOISE,
            verdict=Verdict.AUTHENTIC,
            confidence=0.60,
            reasoning="r2",
        ),
        "fatformer": AgentResponse(
            agent_name="fat",
            role=AgentRole.FATFORMER,
            verdict=Verdict.MANIPULATED,
            confidence=0.80,
            reasoning="r3",
        ),
        "spatial": AgentResponse(
            agent_name="spatial",
            role=AgentRole.SPATIAL,
            verdict=Verdict.MANIPULATED,
            confidence=0.65,
            reasoning="r4",
        ),
    }

    expected = manager.consensus_engine.aggregate(
        responses,
        manager.agent_trust,
        algorithm=manager.consensus_algorithm,
    )
    result, debate_result, consensus_info, debate_history = manager._run_consensus_pipeline(responses)

    assert debate_result is None
    assert debate_history == []
    assert result.final_verdict == expected.final_verdict
    assert result.confidence == pytest.approx(expected.confidence)
    assert consensus_info["final_verdict"] == expected.final_verdict.value
    assert consensus_info["algorithm"] == expected.algorithm_used


def test_maifs_manager_consensus_settings_are_aligned():
    """
    MAIFS and ManagerAgent must share the same consensus/debate runtime knobs.
    """
    maifs = MAIFS(
        enable_debate=False,
        debate_threshold=0.42,
        consensus_algorithm="avga",
        device="cpu",
    )

    assert maifs.enable_debate is maifs.manager.enable_debate
    assert maifs.debate_threshold == pytest.approx(maifs.manager.debate_threshold)
    assert maifs.consensus_algorithm == maifs.manager.consensus_algorithm


def test_manager_auto_algorithm_matches_direct_auto_aggregate():
    """
    consensus_algorithm='auto'인 경우 manager 파이프라인이
    COBRA 자동 선택과 동일하게 동작해야 한다.
    """
    manager = ManagerAgent(use_llm=False, enable_debate=False, consensus_algorithm="auto")
    responses = {
        "frequency": AgentResponse(
            agent_name="freq",
            role=AgentRole.FREQUENCY,
            verdict=Verdict.AI_GENERATED,
            confidence=0.80,
            reasoning="r1",
        ),
        "noise": AgentResponse(
            agent_name="noise",
            role=AgentRole.NOISE,
            verdict=Verdict.AUTHENTIC,
            confidence=0.65,
            reasoning="r2",
        ),
        "fatformer": AgentResponse(
            agent_name="fat",
            role=AgentRole.FATFORMER,
            verdict=Verdict.AI_GENERATED,
            confidence=0.88,
            reasoning="r3",
        ),
        "spatial": AgentResponse(
            agent_name="spatial",
            role=AgentRole.SPATIAL,
            verdict=Verdict.MANIPULATED,
            confidence=0.62,
            reasoning="r4",
        ),
    }

    expected = manager.consensus_engine.aggregate(
        responses,
        manager.agent_trust,
        algorithm=None,
    )
    result, _, _, _ = manager._run_consensus_pipeline(responses)
    assert result.algorithm_used == expected.algorithm_used
    assert result.final_verdict == expected.final_verdict
    assert result.confidence == pytest.approx(expected.confidence)


def test_resolve_trust_supports_metrics_derived_base_and_yaml_override():
    """
    trust는 메트릭 기반 파생값을 base로 사용하고, yaml override로 일부 덮을 수 있어야 한다.
    """
    metrics = {
        "frequency": {"macro_f1": 0.90, "balanced_accuracy": 0.92, "ece": 0.05},
        "noise": {"macro_f1": 0.40, "balanced_accuracy": 0.45, "ece": 0.20},
        "fatformer": {"macro_f1": 0.88, "balanced_accuracy": 0.87, "ece": 0.03},
        "spatial": {"macro_f1": 0.72, "balanced_accuracy": 0.70, "ece": 0.10},
    }
    derived = derive_trust_from_metrics(metrics)
    merged = resolve_trust({"noise": 0.77}, metrics_override=metrics)

    assert 0.0 <= derived["frequency"] <= 1.0
    assert derived["frequency"] > derived["noise"]
    assert merged["noise"] == pytest.approx(0.77)
