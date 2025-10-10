"""Tests for general utility modifiers."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from eflips.model import (
    Rotation,
    Scenario,
    Trip,
    Route,
    Station,
    VehicleType,
    TripType,
    Line,
    AssocRouteStation,
    Temperatures,
)
from sqlalchemy.orm import Session

from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData, AddTemperatures


class TestRemoveUnusedData:
    """Test suite for RemoveUnusedData modifier."""

    @pytest.fixture
    def scenario_with_unused_data(self, db_session: Session) -> Scenario:
        """Create a test scenario with unused routes, lines, and stations."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        # Create stations
        station_used_1 = Station(
            name="Used Station 1",
            name_short="USED1",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        station_used_2 = Station(
            name="Used Station 2",
            name_short="USED2",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        # Station that will have no routes
        station_unused = Station(
            name="Unused Station",
            name_short="UNUSED",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([station_used_1, station_used_2, station_unused])
        db_session.flush()

        # Create a vehicle type
        vt = VehicleType(
            name="Test Bus",
            scenario_id=scenario.id,
            name_short="TB",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        db_session.add(vt)
        db_session.flush()

        # Create lines
        line_with_routes = Line(
            name="Line with Routes",
            name_short="L1",
            scenario_id=scenario.id,
        )
        line_without_routes = Line(
            name="Line without Routes",
            name_short="L2",
            scenario_id=scenario.id,
        )
        db_session.add_all([line_with_routes, line_without_routes])
        db_session.flush()

        # Create routes
        # Route with trips (will be kept)
        route_with_trips = Route(
            name="Route with Trips",
            name_short="R1",
            scenario_id=scenario.id,
            line=line_with_routes,
            departure_station=station_used_1,
            arrival_station=station_used_2,
            distance=5000,
        )
        # Route without trips (will be removed)
        route_without_trips = Route(
            name="Route without Trips",
            name_short="R2",
            scenario_id=scenario.id,
            line=line_with_routes,
            departure_station=station_used_1,
            arrival_station=station_used_2,
            distance=3000,
        )
        db_session.add_all([route_with_trips, route_without_trips])
        db_session.flush()

        # Add AssocRouteStation for the route without trips
        assoc_1 = AssocRouteStation(
            scenario_id=scenario.id,
            route=route_without_trips,
            station=station_used_1,
            elapsed_distance=0,
            location=None,
        )
        assoc_2 = AssocRouteStation(
            scenario_id=scenario.id,
            route=route_without_trips,
            station=station_used_2,
            elapsed_distance=3000,
            location=None,
        )
        db_session.add_all([assoc_1, assoc_2])
        db_session.flush()

        # Create a rotation with trips for the route with trips
        rotation = Rotation(
            name="Test Rotation",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation)
        db_session.flush()

        trip = Trip(
            rotation=rotation,
            route=route_with_trips,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 8, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 8, 30, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add(trip)

        db_session.commit()
        return scenario

    def test_remove_unused_data_basic(
        self, temp_db: Path, scenario_with_unused_data, db_session: Session
    ):
        """Test RemoveUnusedData modifier removes unused routes, lines, and stations."""
        modifier = RemoveUnusedData()

        # Count initial objects
        initial_routes = db_session.query(Route).count()
        initial_lines = db_session.query(Line).count()
        initial_stations = db_session.query(Station).count()

        assert initial_routes == 2
        assert initial_lines == 2
        assert initial_stations == 3

        # Run modifier
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Check that unused objects were removed
        # Should remove: 1 route (without trips), 1 line (without routes), 1 station (not in any route)
        remaining_routes = db_session.query(Route).all()
        remaining_lines = db_session.query(Line).all()
        remaining_stations = db_session.query(Station).all()

        assert len(remaining_routes) == 1
        assert remaining_routes[0].name == "Route with Trips"

        assert len(remaining_lines) == 1
        assert remaining_lines[0].name == "Line with Routes"

        assert len(remaining_stations) == 2
        station_names = {s.name for s in remaining_stations}
        assert "Used Station 1" in station_names
        assert "Used Station 2" in station_names
        assert "Unused Station" not in station_names

    def test_remove_unused_data_removes_assoc_route_stations(
        self, temp_db: Path, scenario_with_unused_data, db_session: Session
    ):
        """Test that AssocRouteStation entries are removed with unused routes."""
        modifier = RemoveUnusedData()

        # Count initial AssocRouteStation entries
        initial_assoc_count = db_session.query(AssocRouteStation).count()
        assert initial_assoc_count == 2  # Added to route_without_trips

        # Run modifier
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Check that AssocRouteStation entries were removed
        remaining_assoc = db_session.query(AssocRouteStation).count()
        assert remaining_assoc == 0  # All should be removed with the unused route

    def test_remove_unused_data_preserves_used_objects(
        self, temp_db: Path, scenario_with_unused_data, db_session: Session
    ):
        """Test that used routes, lines, and stations are preserved."""
        modifier = RemoveUnusedData()

        # Get IDs of objects that should be kept
        route_with_trips = db_session.query(Route).filter(Route.name == "Route with Trips").one()
        line_with_routes = db_session.query(Line).filter(Line.name == "Line with Routes").one()
        used_station_1 = db_session.query(Station).filter(Station.name_short == "USED1").one()
        used_station_2 = db_session.query(Station).filter(Station.name_short == "USED2").one()

        route_id = route_with_trips.id
        line_id = line_with_routes.id
        station_1_id = used_station_1.id
        station_2_id = used_station_2.id

        # Run modifier
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Verify objects still exist
        assert db_session.query(Route).filter(Route.id == route_id).count() == 1
        assert db_session.query(Line).filter(Line.id == line_id).count() == 1
        assert db_session.query(Station).filter(Station.id == station_1_id).count() == 1
        assert db_session.query(Station).filter(Station.id == station_2_id).count() == 1

    def test_remove_unused_data_with_no_unused_objects(self, temp_db: Path, db_session: Session):
        """Test RemoveUnusedData when there are no unused objects."""
        # Create a minimal scenario with everything in use
        scenario = Scenario(name="Clean Scenario", name_short="CLEAN")
        db_session.add(scenario)
        db_session.flush()

        station1 = Station(
            name="Station 1",
            name_short="S1",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        station2 = Station(
            name="Station 2",
            name_short="S2",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([station1, station2])
        db_session.flush()

        vt = VehicleType(
            name="Test Bus",
            scenario_id=scenario.id,
            name_short="TB",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        db_session.add(vt)
        db_session.flush()

        line = Line(
            name="Test Line",
            name_short="TL",
            scenario_id=scenario.id,
        )
        db_session.add(line)
        db_session.flush()

        route = Route(
            name="Test Route",
            name_short="TR",
            scenario_id=scenario.id,
            line=line,
            departure_station=station1,
            arrival_station=station2,
            distance=5000,
        )
        db_session.add(route)
        db_session.flush()

        rotation = Rotation(
            name="Test Rotation",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation)
        db_session.flush()

        trip = Trip(
            rotation=rotation,
            route=route,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 8, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 8, 30, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add(trip)
        db_session.commit()

        # Count objects before
        routes_before = db_session.query(Route).count()
        lines_before = db_session.query(Line).count()
        stations_before = db_session.query(Station).count()

        # Run modifier
        modifier = RemoveUnusedData()
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Count objects after - should be unchanged
        routes_after = db_session.query(Route).count()
        lines_after = db_session.query(Line).count()
        stations_after = db_session.query(Station).count()

        assert routes_before == routes_after == 1
        assert lines_before == lines_after == 1
        assert stations_before == stations_after == 2

    def test_remove_unused_data_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that having multiple scenarios raises an error."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = RemoveUnusedData()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(session=db_session, params={})

    def test_document_params(self):
        """Test that document_params returns empty dict (no parameters)."""
        modifier = RemoveUnusedData()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 0  # No parameters for this modifier

    def test_remove_unused_data_cascade_deletion(self, temp_db: Path, db_session: Session):
        """Test that removing a line also removes all its unused routes."""
        # Create scenario
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        station1 = Station(
            name="Station 1",
            name_short="S1",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        station2 = Station(
            name="Station 2",
            name_short="S2",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([station1, station2])
        db_session.flush()

        # Create a line with multiple routes (all without trips)
        line = Line(
            name="Line with Unused Routes",
            name_short="L1",
            scenario_id=scenario.id,
        )
        db_session.add(line)
        db_session.flush()

        route1 = Route(
            name="Unused Route 1",
            name_short="UR1",
            scenario_id=scenario.id,
            line=line,
            departure_station=station1,
            arrival_station=station2,
            distance=5000,
        )
        route2 = Route(
            name="Unused Route 2",
            name_short="UR2",
            scenario_id=scenario.id,
            line=line,
            departure_station=station1,
            arrival_station=station2,
            distance=3000,
        )
        db_session.add_all([route1, route2])
        db_session.commit()

        # Verify initial state
        assert db_session.query(Line).count() == 1
        assert db_session.query(Route).count() == 2

        # Run modifier
        modifier = RemoveUnusedData()
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # All routes should be removed (no trips)
        # Then the line should be removed (no routes)
        assert db_session.query(Route).count() == 0
        assert db_session.query(Line).count() == 0


class TestAddTemperatures:
    """Test suite for AddTemperatures modifier."""

    def test_add_temperatures_with_defaults(self, temp_db: Path, db_session: Session):
        """Test AddTemperatures with default parameters."""
        # Create a scenario
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.commit()

        # Verify no temperatures exist initially
        initial_temps = db_session.query(Temperatures).count()
        assert initial_temps == 0

        # Run modifier with defaults
        modifier = AddTemperatures()
        with pytest.warns(UserWarning, match="Using default temperature"):
            modifier.modify(session=db_session, params={})
        db_session.commit()

        # Check that temperature was added
        temps = db_session.query(Temperatures).all()
        assert len(temps) == 1
        assert temps[0].scenario_id == scenario.id
        assert temps[0].name == "-12.0 °C"
        assert temps[0].use_only_time is False
        assert len(temps[0].datetimes) == 2
        assert len(temps[0].data) == 2
        assert temps[0].data[0] == -12.0
        assert temps[0].data[1] == -12.0

        # Check that datetimes use UTC and are min/max
        assert temps[0].datetimes[0].tzinfo.key == "UTC"
        assert temps[0].datetimes[1].tzinfo.key == "UTC"

    def test_add_temperatures_with_custom_temperature(self, temp_db: Path, db_session: Session):
        """Test AddTemperatures with custom temperature."""
        # Create a scenario
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.commit()

        # Run modifier with custom temperature
        modifier = AddTemperatures()
        modifier.modify(session=db_session, params={"AddTemperatures.temperature_celsius": 25.0})
        db_session.commit()

        # Check that temperature was added with custom value
        temps = db_session.query(Temperatures).all()
        assert len(temps) == 1
        assert temps[0].name == "25.0 °C"
        assert temps[0].data[0] == 25.0
        assert temps[0].data[1] == 25.0

    def test_add_temperatures_validation_invalid_type(self, temp_db: Path, db_session: Session):
        """Test that invalid temperature type raises error."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.commit()

        modifier = AddTemperatures()

        with pytest.raises(ValueError, match="Temperature must be a number"):
            modifier.modify(
                session=db_session,
                params={"AddTemperatures.temperature_celsius": "not a number"},
            )

    def test_add_temperatures_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that multiple scenarios raises an error."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = AddTemperatures()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(session=db_session, params={})

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = AddTemperatures()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 1
        assert "AddTemperatures.temperature_celsius" in docs
