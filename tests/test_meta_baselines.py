"""
Meta baseline regression tests.
"""
from __future__ import annotations

import numpy as np

from src.meta.baselines import COBRABaseline
from src.meta.simulator import SimulatedOutput


def _sample_case() -> SimulatedOutput:
    return SimulatedOutput(
        true_label="ai_generated",
        agent_verdicts={
            "frequency": "ai_generated",
            "noise": "authentic",
            "fatformer": "ai_generated",
            "spatial": "manipulated",
        },
        agent_confidences={
            "frequency": 0.90,
            "noise": 0.60,
            "fatformer": 0.88,
            "spatial": 0.55,
        },
        sub_type="gan",
    )


def test_cobra_baseline_honors_algorithm_parameter():
    sample = _sample_case()

    rot = COBRABaseline(algorithm="rot")
    drwa = COBRABaseline(algorithm="drwa")
    avga = COBRABaseline(algorithm="avga")

    assert rot._aggregate_sample(sample).algorithm_used == "RoT"
    assert drwa._aggregate_sample(sample).algorithm_used == "DRWA"
    assert avga._aggregate_sample(sample).algorithm_used == "AVGA"


def test_cobra_baseline_predict_proba_shape_and_normalization():
    sample = _sample_case()
    cobra = COBRABaseline(algorithm="drwa")

    proba = cobra.predict_proba([sample, sample])
    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

