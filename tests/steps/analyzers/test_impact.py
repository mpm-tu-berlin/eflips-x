"""Regression tests for the eflips-impact configurators and analyzers.

These are regression tests rather than specifications: they pin down the behaviour
the BVG flow currently relies on, so that changes in eflips-x or eflips-impact that
would silently alter TCO/LCA results show up as failures.

The three behaviours most worth protecting, because each one fails *quietly*:

- the impact JSONs are content-hashed into the cache key, so editing one re-runs
  the step instead of serving a stale result;
- the configurators raise when eflips-impact warn-and-skips an entity, instead of
  letting the calculators treat the missing parameters as zero;
- the analyzers warn when the annualisation factor is auto-detected, and honour it
  when pinned.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objs as go
import pytest
from eflips.model import (
    BatteryType,
    ChargingPointType,
    EnergySource,
    Scenario,
    VehicleType,
)
from sqlalchemy.orm import Session

from eflips.x.framework import PipelineContext
from eflips.x.steps.analyzers.output_analyzers import (
    LCAAnalyzer,
    TCOAnalyzer,
    merge_lca_results,
    merge_tco_results,
)
from eflips.x.steps.modifiers.general_utilities import (
    CompleteFleet,
    LCAConfigurator,
    TCOConfigurator,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
REPO_LCA_JSON = PROJECT_ROOT / "data" / "input" / "impact" / "lca.json"

# The single vehicle type created by tests.util.multi_depot_scenario.
VT_NAME_SHORT = "EB12"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def impact_db(simulated_db_path: Path, tmp_path: Path) -> Path:
    """Writable copy of the shared simulated DB, with an explicit energy source.

    ``multi_depot_scenario`` does not name an energy source, so its vehicle type
    relies on the ``BATTERY_ELECTRIC`` column default. Every eflips-impact entry
    point branches on that value, so state it here rather than letting these tests
    depend on a default defined in eflips-model.
    """
    import eflips.model

    db_copy = tmp_path / "impact.db"
    shutil.copy2(simulated_db_path, db_copy)

    engine = eflips.model.create_engine(f"sqlite:///{db_copy.absolute().as_posix()}")
    session = Session(engine)
    try:
        for vehicle_type in session.query(VehicleType).all():
            vehicle_type.energy_source = EnergySource.BATTERY_ELECTRIC
        session.commit()
    finally:
        session.close()
        engine.dispose()

    return db_copy


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def fleet_json(tmp_path: Path) -> Path:
    """Fleet topology matching the depot-only test scenario.

    Only a depot charging point type is declared: the test scenario has no
    opportunity charging, and an unreferenced opportunity type would be left
    without parameters by ``init_tco_params``.
    """
    return _write_json(
        tmp_path / "fleet.json",
        {
            "schema_version": 1,
            "battery_types": [
                {
                    "vehicle_name_short": VT_NAME_SHORT,
                    "specific_mass": 6.0,
                    "chemistry": "lfp",
                }
            ],
            "charging_point_types": [
                {"type": "depot", "name": "Depot Charging Point", "name_short": "DCS"}
            ],
        },
    )


@pytest.fixture
def tco_json(tmp_path: Path) -> Path:
    """TCO parameters covering the test scenario's single vehicle type."""
    return _write_json(
        tmp_path / "tco.json",
        {
            "scenario": {
                "project_duration": 20,
                "interest_rate": 0.04,
                "inflation_rate": 0.02,
                "staff_cost": 25.0,
                "fuel_cost": {"diesel": 1.0, "electricity": 0.1794},
                "vehicle_maint_cost": {"diesel": 0.5, "electricity": 0.35},
                "infra_maint_cost": 1000.0,
                "cost_escalation_rate": {
                    "general": 0.02,
                    "staff": 0.025,
                    "diesel": 0.0,
                    "electricity": 0.038,
                    "insurance": 0.02,
                },
                "insurance": 9693.0,
                "taxes": 278.0,
                "eta_avail": 0.9,
            },
            "vehicle_types": [
                {
                    "name_short": VT_NAME_SHORT,
                    "useful_life": 14,
                    "procurement_cost": 580000.0,
                    "cost_escalation": 0.02,
                    "average_electricity_consumption": 1.48,
                }
            ],
            "battery_types": [
                {
                    "vehicle_name_short": VT_NAME_SHORT,
                    "procurement_cost": 190,
                    "useful_life": 7,
                    "cost_escalation": -0.03,
                }
            ],
            "charging_point_types": [
                {
                    "type": "depot",
                    "procurement_cost": 119899.50,
                    "useful_life": 20,
                    "cost_escalation": 0.02,
                }
            ],
            "charging_infrastructure": [
                {
                    "type": "depot",
                    "procurement_cost": 2397989.95,
                    "useful_life": 20,
                    "cost_escalation": 0.02,
                }
            ],
        },
    )


@pytest.fixture
def lca_overrides_json(tmp_path: Path) -> Path:
    """Per-scenario LCA overrides covering the test scenario's vehicle type."""
    return _write_json(
        tmp_path / "lca_overrides.json",
        {
            "schema_version": 1,
            "year": 2025,
            "vehicle_type_overrides": [
                {
                    "name_short": VT_NAME_SHORT,
                    "motor_rated_power_kw": 200.0,
                    "motor_power_to_weight_ratio_kw_per_kg": 1.5,
                    "vehicle_lifetime_years": 12.0,
                    "average_consumption_kwh_per_km": 1.48,
                    "diesel_consumption_kg_per_km": None,
                }
            ],
            "charging_point_type_overrides": [
                {
                    "type": "depot",
                    "infrastructure_lifetime_years": 20.0,
                    "foundation_volume_per_point_m3": 3.96,
                }
            ],
        },
    )


@pytest.fixture
def lca_json(tmp_path: Path) -> Path:
    """The repository's openLCA emission-factor export.

    Copied rather than referenced so a test can mutate it without touching the
    checked-in file. Using the real file also keeps it parseable-by-construction.
    """
    destination = tmp_path / "lca.json"
    shutil.copy2(REPO_LCA_JSON, destination)
    return destination


@pytest.fixture
def impact_session(impact_db: Path):
    """Session onto the per-test impact database."""
    import eflips.model

    engine = eflips.model.create_engine(f"sqlite:///{impact_db.absolute().as_posix()}")
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def configured_session(
    impact_session: Session,
    fleet_json: Path,
    tco_json: Path,
    lca_overrides_json: Path,
    lca_json: Path,
) -> Session:
    """Session with fleet topology plus TCO and LCA parameters applied."""
    CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})
    TCOConfigurator(tco_json=tco_json).modify(impact_session, {})
    LCAConfigurator(lca_json=lca_json, lca_overrides_json=lca_overrides_json).modify(
        impact_session, {}
    )
    impact_session.flush()
    return impact_session


# ---------------------------------------------------------------------------
# CompleteFleet
# ---------------------------------------------------------------------------


class TestCompleteFleet:
    def test_creates_and_assigns_fleet_topology(
        self, impact_session: Session, fleet_json: Path
    ) -> None:
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})
        impact_session.flush()

        scenario = impact_session.query(Scenario).one()
        battery_types = (
            impact_session.query(BatteryType).filter(BatteryType.scenario_id == scenario.id).all()
        )
        assert len(battery_types) == 1
        assert battery_types[0].chemistry == "lfp"

        charging_point_types = (
            impact_session.query(ChargingPointType)
            .filter(ChargingPointType.scenario_id == scenario.id)
            .all()
        )
        assert len(charging_point_types) == 1

        vehicle_type = (
            impact_session.query(VehicleType).filter(VehicleType.name_short == VT_NAME_SHORT).one()
        )
        assert vehicle_type.battery_type_id == battery_types[0].id

    def test_is_idempotent(self, impact_session: Session, fleet_json: Path) -> None:
        """Re-running rebuilds the topology rather than accumulating rows."""
        step = CompleteFleet(fleet_json=fleet_json)
        step.modify(impact_session, {})
        impact_session.flush()
        step.modify(impact_session, {})
        impact_session.flush()

        scenario = impact_session.query(Scenario).one()
        assert (
            impact_session.query(BatteryType)
            .filter(BatteryType.scenario_id == scenario.id)
            .count()
            == 1
        )

    def test_raises_when_vehicle_type_absent_from_fleet_json(
        self, impact_session: Session, tmp_path: Path
    ) -> None:
        """A fleet JSON that omits a vehicle type must fail, not silently no-op."""
        empty_fleet = _write_json(
            tmp_path / "empty_fleet.json",
            {"schema_version": 1, "battery_types": [], "charging_point_types": []},
        )

        with pytest.raises(ValueError, match="no BatteryType was assigned"):
            CompleteFleet(fleet_json=empty_fleet).modify(impact_session, {})

    def test_allow_incomplete_parameters_downgrades_to_warning(
        self, impact_session: Session, tmp_path: Path
    ) -> None:
        empty_fleet = _write_json(
            tmp_path / "empty_fleet.json",
            {"schema_version": 1, "battery_types": [], "charging_point_types": []},
        )
        step = CompleteFleet(fleet_json=empty_fleet)

        # Must not raise.
        step.modify(impact_session, {"CompleteFleet.allow_incomplete_parameters": True})

    def test_missing_file_raises(self, impact_session: Session, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CompleteFleet(fleet_json=tmp_path / "nope.json").modify(impact_session, {})


# ---------------------------------------------------------------------------
# TCOConfigurator / LCAConfigurator
# ---------------------------------------------------------------------------


class TestTCOConfigurator:
    def test_writes_tco_parameters(
        self, impact_session: Session, fleet_json: Path, tco_json: Path
    ) -> None:
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})
        TCOConfigurator(tco_json=tco_json).modify(impact_session, {})
        impact_session.flush()

        scenario = impact_session.query(Scenario).one()
        assert scenario.tco_parameters is not None
        assert scenario.tco_parameters["project_duration"] == 20

        vehicle_type = (
            impact_session.query(VehicleType).filter(VehicleType.name_short == VT_NAME_SHORT).one()
        )
        assert vehicle_type.tco_parameters is not None
        assert vehicle_type.tco_parameters["procurement_cost"] == 580000.0

        battery_type = impact_session.query(BatteryType).one()
        assert battery_type.tco_parameters is not None

    def test_raises_when_vehicle_type_missing_from_json(
        self, impact_session: Session, fleet_json: Path, tco_json: Path, tmp_path: Path
    ) -> None:
        """A name_short typo must fail loudly.

        eflips-model gives ``tco_parameters`` a server-side default, so a vehicle type
        the JSON does not mention is costed at placeholder values rather than left
        empty -- the failure is invisible in the output.
        """
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})

        payload = json.loads(tco_json.read_text(encoding="utf-8"))
        payload["vehicle_types"][0]["name_short"] = "TYPO"
        typo_json = _write_json(tmp_path / "tco_typo.json", payload)

        with pytest.raises(ValueError, match="does not cover"):
            TCOConfigurator(tco_json=typo_json).modify(impact_session, {})

    def test_entries_without_a_matching_vehicle_type_are_tolerated(
        self, impact_session: Session, fleet_json: Path, tco_json: Path, tmp_path: Path
    ) -> None:
        """One parameter file is shared across scenarios with different fleets.

        The BVG flow reuses a single tco.json for the electric and diesel scenarios,
        so surplus entries must not be an error.
        """
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})

        payload = json.loads(tco_json.read_text(encoding="utf-8"))
        surplus = dict(payload["vehicle_types"][0])
        surplus["name_short"] = "NOT_IN_THIS_SCENARIO"
        payload["vehicle_types"].append(surplus)
        shared_json = _write_json(tmp_path / "tco_shared.json", payload)

        TCOConfigurator(tco_json=shared_json).modify(impact_session, {})

    def test_allow_incomplete_parameters_downgrades_to_warning(
        self, impact_session: Session, fleet_json: Path, tco_json: Path, tmp_path: Path
    ) -> None:
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})

        payload = json.loads(tco_json.read_text(encoding="utf-8"))
        payload["vehicle_types"][0]["name_short"] = "TYPO"
        typo_json = _write_json(tmp_path / "tco_typo.json", payload)

        TCOConfigurator(tco_json=typo_json).modify(
            impact_session, {"TCOConfigurator.allow_incomplete_parameters": True}
        )

    def test_missing_file_raises(self, impact_session: Session, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TCOConfigurator(tco_json=tmp_path / "nope.json").modify(impact_session, {})


class TestLCAConfigurator:
    def test_writes_lca_parameters(
        self,
        impact_session: Session,
        fleet_json: Path,
        lca_json: Path,
        lca_overrides_json: Path,
    ) -> None:
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})
        LCAConfigurator(lca_json=lca_json, lca_overrides_json=lca_overrides_json).modify(
            impact_session, {}
        )
        impact_session.flush()

        vehicle_type = (
            impact_session.query(VehicleType).filter(VehicleType.name_short == VT_NAME_SHORT).one()
        )
        assert vehicle_type.lca_parameters is not None
        assert impact_session.query(BatteryType).one().lca_parameters is not None
        assert impact_session.query(ChargingPointType).one().lca_parameters is not None

    def test_raises_when_overrides_omit_a_vehicle_type(
        self,
        impact_session: Session,
        fleet_json: Path,
        lca_json: Path,
        lca_overrides_json: Path,
        tmp_path: Path,
    ) -> None:
        """init_lca_params skips the *whole scenario* here; that must not pass silently."""
        CompleteFleet(fleet_json=fleet_json).modify(impact_session, {})

        payload = json.loads(lca_overrides_json.read_text(encoding="utf-8"))
        payload["vehicle_type_overrides"] = []
        empty_overrides = _write_json(tmp_path / "lca_overrides_empty.json", payload)

        with pytest.raises(ValueError, match="does not cover"):
            LCAConfigurator(lca_json=lca_json, lca_overrides_json=empty_overrides).modify(
                impact_session, {}
            )

    def test_missing_file_raises(self, impact_session: Session, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LCAConfigurator(
                lca_json=tmp_path / "nope.json",
                lca_overrides_json=tmp_path / "also_nope.json",
            ).modify(impact_session, {})


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestImpactJsonCacheInvalidation:
    """The JSON contents must reach the cache key.

    Passing the paths through ``params`` (as an earlier revision did) hashes only
    the path string, so editing a JSON in place silently served cached results.
    """

    @staticmethod
    def _key(step: Any, db_path: Path, tmp_path: Path) -> str:
        context = PipelineContext(work_dir=tmp_path, params={}, current_db=db_path)
        return step.compute_cache_key(context, tmp_path / "out.db")

    def test_fleet_json_content_change_invalidates(
        self, impact_db: Path, fleet_json: Path, tmp_path: Path
    ) -> None:
        step = CompleteFleet(fleet_json=fleet_json)
        before = self._key(step, impact_db, tmp_path)

        payload = json.loads(fleet_json.read_text(encoding="utf-8"))
        payload["battery_types"][0]["specific_mass"] = 7.5
        _write_json(fleet_json, payload)

        assert self._key(step, impact_db, tmp_path) != before

    def test_tco_json_content_change_invalidates(
        self, impact_db: Path, tco_json: Path, tmp_path: Path
    ) -> None:
        step = TCOConfigurator(tco_json=tco_json)
        before = self._key(step, impact_db, tmp_path)

        payload = json.loads(tco_json.read_text(encoding="utf-8"))
        payload["vehicle_types"][0]["procurement_cost"] = 999999.0
        _write_json(tco_json, payload)

        assert self._key(step, impact_db, tmp_path) != before

    def test_lca_overrides_content_change_invalidates(
        self, impact_db: Path, lca_json: Path, lca_overrides_json: Path, tmp_path: Path
    ) -> None:
        step = LCAConfigurator(lca_json=lca_json, lca_overrides_json=lca_overrides_json)
        before = self._key(step, impact_db, tmp_path)

        payload = json.loads(lca_overrides_json.read_text(encoding="utf-8"))
        payload["vehicle_type_overrides"][0]["vehicle_lifetime_years"] = 15.0
        _write_json(lca_overrides_json, payload)

        assert self._key(step, impact_db, tmp_path) != before

    def test_unchanged_json_keeps_key_stable(
        self, impact_db: Path, tco_json: Path, tmp_path: Path
    ) -> None:
        step = TCOConfigurator(tco_json=tco_json)
        assert self._key(step, impact_db, tmp_path) == self._key(step, impact_db, tmp_path)


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------


class TestTCOAnalyzer:
    def test_returns_all_cost_categories(self, configured_session: Session) -> None:
        result = TCOAnalyzer().analyze(
            configured_session,
            {"TCOAnalyzer.scenario_name": "TEST", "TCOAnalyzer.scaling_factor": 52.0},
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["scenario_name"].iloc[0] == "TEST"
        for category in TCOAnalyzer.COST_CATEGORIES:
            assert category in result.columns
        assert result[TCOAnalyzer.COST_CATEGORIES].sum(axis=1).iloc[0] > 0

    def test_warns_when_scaling_factor_auto_detected(
        self, configured_session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            TCOAnalyzer().analyze(configured_session, {"TCOAnalyzer.scenario_name": "TEST"})

        assert "scaling_factor" in caplog.text
        assert "auto-detects" in caplog.text

    def test_no_warning_when_scaling_factor_pinned(
        self, configured_session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            TCOAnalyzer().analyze(
                configured_session,
                {
                    "TCOAnalyzer.scenario_name": "TEST",
                    "TCOAnalyzer.scaling_factor": 52.0,
                },
            )

        assert "auto-detects" not in caplog.text

    def test_scaling_factor_changes_capital_cost_share(self, configured_session: Session) -> None:
        """Capital cost per revenue-km scales with 1/scaling_factor; opex does not.

        This is the reason the factor is worth pinning, so pin the behaviour too.
        """
        base = TCOAnalyzer().analyze(configured_session, {"TCOAnalyzer.scaling_factor": 52.0})
        doubled = TCOAnalyzer().analyze(configured_session, {"TCOAnalyzer.scaling_factor": 104.0})

        assert doubled["VEHICLE"].iloc[0] == pytest.approx(base["VEHICLE"].iloc[0] / 2, rel=1e-6)
        assert doubled["ENERGY"].iloc[0] == pytest.approx(base["ENERGY"].iloc[0], rel=1e-6)

    def test_visualize_returns_figure(self, configured_session: Session) -> None:
        result = TCOAnalyzer().analyze(configured_session, {"TCOAnalyzer.scaling_factor": 52.0})
        assert isinstance(TCOAnalyzer.visualize(result), go.Figure)


class TestLCAAnalyzer:
    def test_returns_both_breakdowns(self, configured_session: Session) -> None:
        result = LCAAnalyzer().analyze(
            configured_session,
            {"LCAAnalyzer.scenario_name": "TEST", "LCAAnalyzer.scaling_factor": 52.0},
        )

        assert len(result) == 1
        assert result["scenario_name"].iloc[0] == "TEST"
        for column in LCAAnalyzer.IMPACT_CATEGORIES + LCAAnalyzer.SCOPE_CATEGORIES:
            assert column in result.columns

    def test_breakdowns_sum_to_the_same_total(self, configured_session: Session) -> None:
        """By type and by scope are two views of one number; keep them consistent."""
        result = LCAAnalyzer().analyze(configured_session, {"LCAAnalyzer.scaling_factor": 52.0})

        by_type = result[LCAAnalyzer.IMPACT_CATEGORIES].sum(axis=1).iloc[0]
        by_scope = result[LCAAnalyzer.SCOPE_CATEGORIES].sum(axis=1).iloc[0]

        assert by_type > 0
        assert by_type == pytest.approx(by_scope, rel=1e-6)

    def test_warns_when_scaling_factor_auto_detected(
        self, configured_session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            LCAAnalyzer().analyze(configured_session, {})

        assert "scaling_factor" in caplog.text
        assert "auto-detects" in caplog.text

    def test_visualizers_produce_figures(self, configured_session: Session) -> None:
        result = LCAAnalyzer().analyze(configured_session, {"LCAAnalyzer.scaling_factor": 52.0})

        for method_name in LCAAnalyzer.VISUALIZERS.values():
            figure = getattr(LCAAnalyzer, method_name)(result)
            assert isinstance(figure, go.Figure)


class TestMergeResults:
    def test_merge_tco_results_concatenates_rows(self, configured_session: Session) -> None:
        analyzer = TCOAnalyzer()
        rows = [
            analyzer.analyze(
                configured_session,
                {"TCOAnalyzer.scenario_name": name, "TCOAnalyzer.scaling_factor": 52.0},
            )
            for name in ("A", "B")
        ]

        merged = merge_tco_results(rows)
        assert list(merged["scenario_name"]) == ["A", "B"]

    def test_merge_lca_results_concatenates_rows(self, configured_session: Session) -> None:
        analyzer = LCAAnalyzer()
        rows = [
            analyzer.analyze(
                configured_session,
                {"LCAAnalyzer.scenario_name": name, "LCAAnalyzer.scaling_factor": 52.0},
            )
            for name in ("A", "B")
        ]

        merged = merge_lca_results(rows)
        assert list(merged["scenario_name"]) == ["A", "B"]
