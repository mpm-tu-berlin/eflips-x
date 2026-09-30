"""Tests for the consumption LUT helpers."""

from datetime import datetime, timedelta

import pytest
from eflips.depot.api import ConsumptionResult  # type: ignore[import-untyped]

from eflips.x.steps.modifiers import consumption_luts
from eflips.x.steps.modifiers.consumption_luts import generate_clamped_consumption_result


def test_generate_clamped_consumption_result(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = datetime(2026, 1, 1, 8, 0)
    timestamps = [t0 + timedelta(minutes=i) for i in range(3)]
    raw = {
        # Ordinary trip: left untouched
        1: ConsumptionResult(-0.05, timestamps, [-0.01, -0.03, -0.05]),
        # Downhill trip with a net energy gain: clamped to zero
        2: ConsumptionResult(0.002, timestamps, [-0.001, 0.001, 0.002]),
        # No timeseries
        3: ConsumptionResult(0.01, None, None),
    }
    monkeypatch.setattr(consumption_luts, "generate_consumption_result", lambda scenario: raw)

    results = generate_clamped_consumption_result(scenario=None)  # type: ignore[arg-type]

    assert results[1].delta_soc_total == -0.05
    assert results[1].delta_soc == [-0.01, -0.03, -0.05]
    assert results[2].delta_soc_total == 0.0
    assert results[2].delta_soc == [-0.001, 0.0, 0.0]
    assert results[2].timestamps == timestamps
    assert results[3].delta_soc_total == 0.0
    assert results[3].delta_soc is None
