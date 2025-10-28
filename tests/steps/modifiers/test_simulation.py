"""Tests for simulation modifiers."""

from pathlib import Path

import pytest
from eflips.depot.api import DepotConfigurationWish
from eflips.model import Scenario, Depot, Area, Process, Station, Rotation
from sqlalchemy.orm import Session

from eflips.x.steps.modifiers.simulation import DepotGenerator
from tests.util import multi_depot_scenario


class TestDepotGenerator:
    """Test suite for DepotGenerator modifier."""

    @pytest.fixture
    def test_scenario(self, db_session: Session) -> Scenario:
        """Create a test scenario with multi-depot network, but delete Depot objects."""
        scenario = multi_depot_scenario(
            db_session,
            num_depots=2,
            lines_per_depot=4,
            trips_per_line=10,
        )

        # Delete the Depot objects created by multi_depot_scenario
        # DepotGenerator should create them
        db_session.query(Depot).filter_by(scenario_id=scenario.id).delete()
        db_session.commit()

        return scenario

    def _get_depot_stations(self, db_session: Session, scenario: Scenario) -> list[Station]:
        """Helper to get depot stations from a scenario (stations where rotations start/end)."""
        # Query all rotations for this scenario
        rotations = db_session.query(Rotation).filter_by(scenario_id=scenario.id).all()

        # Collect unique station IDs where rotations start or end
        depot_station_ids = set()
        for rotation in rotations:
            # Get first and last trip for this rotation
            if rotation.trips:
                first_trip = min(rotation.trips, key=lambda t: t.departure_time)
                last_trip = max(rotation.trips, key=lambda t: t.arrival_time)
                depot_station_ids.add(first_trip.route.departure_station_id)
                depot_station_ids.add(last_trip.route.arrival_station_id)

        # Get the actual station objects
        depot_stations = db_session.query(Station).filter(Station.id.in_(depot_station_ids)).all()
        return depot_stations

    def test_depot_generator_simple_layout(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test DepotGenerator with simple layout (default mode)."""
        # Verify no depot infrastructure exists initially (only depot records)
        initial_areas = db_session.query(Area).count()
        initial_processes = db_session.query(Process).count()
        assert initial_areas == 0
        assert initial_processes == 0

        # Run modifier with simple layout (generate_optimal_depots=False is default)
        modifier = DepotGenerator()
        result = modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.charging_power_kw": 100.0,
            },
        )
        db_session.commit()

        # Verify result
        assert result is None

        # Verify depot infrastructure was created
        areas = db_session.query(Area).all()
        processes = db_session.query(Process).all()

        assert len(areas) > 0, "Should have created depot areas"
        assert len(processes) > 0, "Should have created depot processes"

        # Verify that Depot objects were created
        depots = db_session.query(Depot).filter_by(scenario_id=test_scenario.id).all()
        assert len(depots) > 0, "Should have created Depot objects"

    def test_depot_generator_optimal_layout(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test DepotGenerator with optimal layout generation."""
        # Verify no depot infrastructure exists initially
        initial_areas = db_session.query(Area).count()
        initial_processes = db_session.query(Process).count()
        initial_depots = db_session.query(Depot).count()
        assert initial_areas == 0
        assert initial_processes == 0
        assert initial_depots == 0

        # Run modifier with optimal depot generation
        modifier = DepotGenerator()
        result = modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.generate_optimal_depots": True,
                "DepotGenerator.charging_power_kw": 150.0,
                "DepotGenerator.standard_block_length": 8,
            },
        )
        db_session.commit()

        # Verify result
        assert result is None

        # Verify depot infrastructure was created
        areas = db_session.query(Area).all()
        processes = db_session.query(Process).all()
        depots = db_session.query(Depot).all()

        assert len(areas) > 0, "Should have created depot areas"
        assert len(processes) > 0, "Should have created depot processes"
        assert len(depots) > 0, "Should have created Depot objects"

    def test_depot_generator_with_depot_wishes_auto_generate(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test DepotGenerator with auto-generated depot configuration wishes."""
        # Get depot stations (stations where rotations start/end)
        depot_stations = self._get_depot_stations(db_session, test_scenario)
        assert len(depot_stations) == 2, "Should have 2 depot stations from test_scenario fixture"

        # Create depot configuration wishes for each depot station with auto_generate=True
        depot_wishes = []
        for station in depot_stations:
            wish = DepotConfigurationWish(
                station_id=station.id,
                auto_generate=True,
                default_power=120.0,
                standard_block_length=6,
            )
            depot_wishes.append(wish)

        # Verify no depot infrastructure exists initially
        initial_areas = db_session.query(Area).count()
        initial_depots = db_session.query(Depot).count()
        assert initial_areas == 0
        assert initial_depots == 0

        # Run modifier with depot wishes
        modifier = DepotGenerator()
        result = modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.depot_wishes": depot_wishes,
            },
        )
        db_session.commit()

        # Verify result
        assert result is None

        # Verify depot infrastructure was created
        areas = db_session.query(Area).all()
        depots = db_session.query(Depot).all()
        assert len(areas) > 0, "Should have created depot areas based on wishes"
        assert len(depots) > 0, "Should have created Depot objects"

        # Verify depot count matches station count
        assert len(depots) == len(depot_stations), "Should create one depot per depot station"

    def test_depot_generator_validates_depot_wishes_type(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that depot_wishes must be a list."""
        modifier = DepotGenerator()

        with pytest.raises(TypeError, match="depot_wishes must be a list"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotGenerator.depot_wishes": "not a list",
                },
            )

    def test_depot_generator_validates_depot_wishes_elements(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that depot_wishes elements must be DepotConfigurationWish objects."""
        modifier = DepotGenerator()

        with pytest.raises(TypeError, match="must be a DepotConfigurationWish object"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotGenerator.depot_wishes": [{"not": "a DepotConfigurationWish"}],
                },
            )

    def test_depot_generator_validates_missing_depot_stations(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that depot_wishes must cover all depot stations."""
        # Get depot stations from the scenario
        depot_stations = self._get_depot_stations(db_session, test_scenario)
        assert len(depot_stations) == 2, "Should have 2 depot stations"

        # Create depot wishes for only the first depot (missing the second)
        depot_wishes = [
            DepotConfigurationWish(
                station_id=depot_stations[0].id,
                auto_generate=True,
                default_power=100.0,
                standard_block_length=6,
            ),
        ]

        modifier = DepotGenerator()

        with pytest.raises(ValueError, match="depot_wishes is missing configuration"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotGenerator.depot_wishes": depot_wishes,
                },
            )

    def test_depot_generator_validates_extra_depot_stations(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that depot_wishes cannot include non-depot stations."""
        # Get depot stations from the scenario
        depot_stations = self._get_depot_stations(db_session, test_scenario)
        assert len(depot_stations) == 2, "Should have 2 depot stations"

        # Get a non-depot station (terminus station)
        all_stations = db_session.query(Station).filter_by(scenario_id=test_scenario.id).all()
        depot_station_ids = {s.id for s in depot_stations}
        non_depot_stations = [s for s in all_stations if s.id not in depot_station_ids]
        assert len(non_depot_stations) > 0, "Should have non-depot stations"

        # Create depot wishes including a non-depot station
        depot_wishes = [
            DepotConfigurationWish(
                station_id=depot_stations[0].id,
                auto_generate=True,
                default_power=100.0,
                standard_block_length=6,
            ),
            DepotConfigurationWish(
                station_id=depot_stations[1].id,
                auto_generate=True,
                default_power=100.0,
                standard_block_length=6,
            ),
            DepotConfigurationWish(
                station_id=non_depot_stations[0].id,  # Non-depot station
                auto_generate=True,
                default_power=100.0,
                standard_block_length=6,
            ),
        ]

        modifier = DepotGenerator()

        with pytest.raises(ValueError, match="depot_wishes contains configuration for stations"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotGenerator.depot_wishes": depot_wishes,
                },
            )

    def test_depot_generator_warns_contradictory_params(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session, caplog
    ):
        """Test that contradictory parameters generate warnings."""
        # Get depot stations from the scenario
        depot_stations = self._get_depot_stations(db_session, test_scenario)

        # Create valid depot wishes
        depot_wishes = []
        for station in depot_stations:
            wish = DepotConfigurationWish(
                station_id=station.id,
                auto_generate=True,
                default_power=100.0,
                standard_block_length=6,
            )
            depot_wishes.append(wish)

        modifier = DepotGenerator()

        # Run with both depot_wishes and generate_optimal_depots=True
        # (generate_optimal_depots should be ignored)
        modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.depot_wishes": depot_wishes,
                "DepotGenerator.generate_optimal_depots": True,
            },
        )

        # Check that a warning was logged
        assert any(
            "generate_optimal_depots" in record.message and "ignored" in record.message
            for record in caplog.records
        ), "Should warn about ignored parameter"

    def test_depot_generator_no_scenario_error(self, temp_db: Path, db_session: Session):
        """Test that missing scenario raises an error."""
        modifier = DepotGenerator()

        with pytest.raises(ValueError, match="No scenario found in the database"):
            modifier.modify(session=db_session, params={})

    def test_depot_generator_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that multiple scenarios raise an error."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = DepotGenerator()

        with pytest.raises(
            ValueError, match="Expected exactly one scenario in the database, found 2"
        ):
            modifier.modify(session=db_session, params={})

    def test_depot_generator_default_properties(self):
        """Test default property values."""
        modifier = DepotGenerator()

        assert modifier.default_charging_power_kw == 90.0
        assert modifier.default_standard_block_length == 6

    def test_depot_generator_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = DepotGenerator()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 4
        assert "DepotGenerator.depot_wishes" in docs
        assert "DepotGenerator.generate_optimal_depots" in docs
        assert "DepotGenerator.charging_power_kw" in docs
        assert "DepotGenerator.standard_block_length" in docs

        # Check that descriptions are non-empty
        for key, value in docs.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_depot_generator_deletes_existing_depot(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that existing depot infrastructure is deleted before creating new one."""
        # First, create depot infrastructure
        modifier = DepotGenerator()
        modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.charging_power_kw": 100.0,
            },
        )
        db_session.commit()

        # Count created infrastructure
        first_area_count = db_session.query(Area).count()
        first_process_count = db_session.query(Process).count()
        assert first_area_count > 0
        assert first_process_count > 0

        # Run modifier again with different parameters
        modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.generate_optimal_depots": True,
                "DepotGenerator.charging_power_kw": 150.0,
            },
        )
        db_session.commit()

        # Verify that new infrastructure was created (counts may differ)
        second_area_count = db_session.query(Area).count()
        second_process_count = db_session.query(Process).count()

        # We should still have infrastructure (not zero)
        assert second_area_count > 0
        assert second_process_count > 0

    def test_depot_generator_uses_default_charging_power(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that default charging power is used when not specified."""
        modifier = DepotGenerator()

        # Run without specifying charging_power_kw
        result = modifier.modify(session=db_session, params={})
        db_session.commit()

        # Verify result (should succeed with defaults)
        assert result is None

        # Verify depot infrastructure was created
        areas = db_session.query(Area).all()
        assert len(areas) > 0, "Should have created depot areas with default power"

    def test_depot_generator_uses_default_standard_block_length(
        self, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that default standard_block_length is used when not specified in optimal mode."""
        modifier = DepotGenerator()

        # Run optimal mode without specifying standard_block_length
        result = modifier.modify(
            session=db_session,
            params={
                "DepotGenerator.generate_optimal_depots": True,
            },
        )
        db_session.commit()

        # Verify result (should succeed with defaults)
        assert result is None

        # Verify depot infrastructure was created
        areas = db_session.query(Area).all()
        assert len(areas) > 0, "Should have created depot areas with default block length"
