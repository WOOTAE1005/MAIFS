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
