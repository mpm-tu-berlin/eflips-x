"""Tests for general utility modifiers."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from eflips.model import (
    EnergySource,
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

from eflips.x.steps.modifiers.general_utilities import (
    AddTemperatures,
    CalculateConsumptionScaling,
    CalibrateConsumptionLut,
    CreateDieselVehicleTypes,
    RemoveConsumptionLuts,
    RemoveUnusedData,
    VehicleTypeBlockAssignment,
)


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


class TestRemoveConsumptionLuts:
    """Tests for RemoveConsumptionLuts modifier."""

    def _make_scenario_with_luts(self, db_session: Session):
        from eflips.model import ConsumptionLut, VehicleClass

        scenario = Scenario(name="Test", name_short="T")
        db_session.add(scenario)
        db_session.flush()

        vt = VehicleType(
            name="Bus",
            name_short="B",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            consumption=1.5,
        )
        db_session.add(vt)
        db_session.flush()

        vc = VehicleClass(name="Test Class", name_short="TC", scenario_id=scenario.id)
        db_session.add(vc)
        vt.vehicle_classes.append(vc)
        db_session.flush()

        lut = ConsumptionLut(
            name="Test LUT",
            scenario_id=scenario.id,
            vehicle_class_id=vc.id,
            columns=["incline", "temp", "loading", "speed"],
            data_points=[[0.0, 20.0, 0.5, 50.0]],
            values=[1.5],
        )
        db_session.add(lut)
        db_session.commit()
        return scenario

    def test_luts_deleted(self, temp_db: Path, db_session: Session):
        from eflips.model import ConsumptionLut

        self._make_scenario_with_luts(db_session)
        assert db_session.query(ConsumptionLut).count() == 1
        RemoveConsumptionLuts().modify(db_session, {})
        db_session.commit()
        assert db_session.query(ConsumptionLut).count() == 0

    def test_vehicle_classes_deleted(self, temp_db: Path, db_session: Session):
        from eflips.model import VehicleClass

        self._make_scenario_with_luts(db_session)
        assert db_session.query(VehicleClass).count() == 1
        RemoveConsumptionLuts().modify(db_session, {})
        db_session.commit()
        assert db_session.query(VehicleClass).count() == 0

    def test_minimal_consumption_applied(self, temp_db: Path, db_session: Session):
        self._make_scenario_with_luts(db_session)
        RemoveConsumptionLuts().modify(
            db_session, {"RemoveConsumptionLuts.minimal_consumption": 0.01}
        )
        db_session.commit()
        for vt in db_session.query(VehicleType).all():
            assert vt.consumption == pytest.approx(0.01)

    def test_raises_on_nonpositive_minimal_consumption(self, temp_db: Path, db_session: Session):
        self._make_scenario_with_luts(db_session)
        with pytest.raises(ValueError, match="minimal_consumption must be positive"):
            RemoveConsumptionLuts().modify(
                db_session, {"RemoveConsumptionLuts.minimal_consumption": 0.0}
            )

    def test_document_params(self):
        docs = RemoveConsumptionLuts().document_params()
        assert isinstance(docs, dict)
        assert "RemoveConsumptionLuts.minimal_consumption" in docs


class TestCalculateConsumptionScaling:
    """Tests for CalculateConsumptionScaling modifier — validation paths only."""

    def test_raises_when_quarterly_consumption_wrong_length(
        self, temp_db: Path, db_session: Session
    ):
        scenario = Scenario(name="Test", name_short="T")
        db_session.add(scenario)
        db_session.commit()

        with pytest.raises(ValueError, match="real_quarterly_consumption must have exactly 4"):
            CalculateConsumptionScaling().modify(
                db_session,
                {"CalculateConsumptionScaling.real_quarterly_consumption": [1.0, 2.0, 3.0]},
            )

    def test_raises_when_multiple_scenarios(self, temp_db: Path, db_session: Session):
        db_session.add(Scenario(name="S1", name_short="S1"))
        db_session.add(Scenario(name="S2", name_short="S2"))
        db_session.commit()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            CalculateConsumptionScaling().modify(db_session, {})

    def test_skips_gracefully_when_no_luts(self, temp_db: Path, db_session: Session):
        scenario = Scenario(name="Test", name_short="T")
        db_session.add(scenario)
        db_session.flush()

        vt = VehicleType(
            name="Bus",
            name_short="B",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            consumption=1.5,
        )
        db_session.add(vt)
        db_session.commit()

        # Should not raise — just log a warning and return
        CalculateConsumptionScaling().modify(db_session, {})

    def test_document_params(self):
        docs = CalculateConsumptionScaling().document_params()
        assert isinstance(docs, dict)
        assert "CalculateConsumptionScaling.real_quarterly_consumption" in docs


class TestCreateDieselVehicleTypes:
    """Test suite for the CreateDieselVehicleTypes modifier."""

    @pytest.fixture
    def scenario_with_electric_types(self, db_session: Session) -> Scenario:
        """Scenario with two battery-electric vehicle types and one rotation each."""
        scenario = Scenario(name="Diesel Test", name_short="DT")
        db_session.add(scenario)
        db_session.flush()

        for name, short in (("Electric Bus 12m", "EN"), ("Electric Bus 18m", "GN")):
            db_session.add(
                VehicleType(
                    scenario_id=scenario.id,
                    name=name,
                    name_short=short,
                    battery_capacity=350.0,
                    battery_capacity_reserve=0.0,
                    charging_curve=[[0, 150], [1, 150]],
                    opportunity_charging_capable=True,
                    minimum_charging_power=10,
                    charging_efficiency=0.95,
                    empty_mass=10000,
                    allowed_mass=20000,
                    consumption=1.2,
                    energy_source=EnergySource.BATTERY_ELECTRIC,
                    length=12.0,
                    width=2.5,
                    height=3.5,
                )
            )
        db_session.flush()
        return scenario

    def test_creates_one_diesel_type_per_electric_type(
        self, temp_db: Path, db_session: Session, scenario_with_electric_types: Scenario
    ):
        CreateDieselVehicleTypes().modify(db_session, {})
        db_session.flush()

        diesel_types = (
            db_session.query(VehicleType)
            .filter(VehicleType.name_short.startswith("Diesel "))
            .all()
        )
        assert {vt.name_short for vt in diesel_types} == {"Diesel EN", "Diesel GN"}
        for diesel_type in diesel_types:
            assert diesel_type.energy_source == EnergySource.DIESEL
            assert diesel_type.consumption == CreateDieselVehicleTypes.DIESEL_CONSUMPTION

    def test_copies_dimensions_needed_by_depot_layout(
        self, temp_db: Path, db_session: Session, scenario_with_electric_types: Scenario
    ):
        """DepotGenerator's optimal-layout mode requires length/width/height."""
        CreateDieselVehicleTypes().modify(db_session, {})
        db_session.flush()

        source = db_session.query(VehicleType).filter(VehicleType.name_short == "EN").one()
        diesel = db_session.query(VehicleType).filter(VehicleType.name_short == "Diesel EN").one()
        assert (diesel.length, diesel.width, diesel.height) == (
            source.length,
            source.width,
            source.height,
        )

    def test_is_idempotent(
        self, temp_db: Path, db_session: Session, scenario_with_electric_types: Scenario
    ):
        step = CreateDieselVehicleTypes()
        step.modify(db_session, {})
        db_session.flush()
        step.modify(db_session, {})
        db_session.flush()

        assert (
            db_session.query(VehicleType).filter(VehicleType.name_short == "Diesel EN").count()
            == 1
        )

    def test_covers_vehicle_types_that_never_set_energy_source(
        self, temp_db: Path, db_session: Session
    ):
        """energy_source is NOT NULL and defaults to BATTERY_ELECTRIC.

        Vehicle types created without naming it are battery-electric already, so they
        must still get diesel counterparts.
        """
        scenario = Scenario(name="Implicit Source", name_short="IS")
        db_session.add(scenario)
        db_session.flush()
        db_session.add(
            VehicleType(
                scenario_id=scenario.id,
                name="Unspecified Bus",
                name_short="UB",
                battery_capacity=350.0,
                battery_capacity_reserve=0.0,
                charging_curve=[[0, 150], [1, 150]],
                opportunity_charging_capable=False,
                minimum_charging_power=10,
                charging_efficiency=0.95,
                empty_mass=10000,
                allowed_mass=20000,
                consumption=1.2,
            )
        )
        db_session.flush()

        CreateDieselVehicleTypes().modify(db_session, {})
        db_session.flush()

        assert (
            db_session.query(VehicleType).filter(VehicleType.name_short == "Diesel UB").count()
            == 1
        )

    def test_document_params(self):
        assert CreateDieselVehicleTypes.document_params() == {}


class TestVehicleTypeBlockAssignment:
    """Test suite for the VehicleTypeBlockAssignment modifier."""

    @pytest.fixture
    def scenario_with_diesel_types(self, db_session: Session) -> Scenario:
        """Scenario with electric types, diesel counterparts, and two rotations."""
        scenario = Scenario(name="Assignment Test", name_short="AT")
        db_session.add(scenario)
        db_session.flush()

        electric = VehicleType(
            scenario_id=scenario.id,
            name="Electric Bus 12m",
            name_short="EN",
            battery_capacity=350.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            charging_efficiency=0.95,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.2,
            energy_source=EnergySource.BATTERY_ELECTRIC,
        )
        db_session.add(electric)
        db_session.flush()

        for index in range(2):
            db_session.add(
                Rotation(
                    scenario_id=scenario.id,
                    vehicle_type_id=electric.id,
                    allow_opportunity_charging=False,
                    name=f"Rotation {index}",
                )
            )
        db_session.flush()

        CreateDieselVehicleTypes().modify(db_session, {})
        db_session.flush()
        return scenario

    def test_reassigns_all_rotations_by_default(
        self, temp_db: Path, db_session: Session, scenario_with_diesel_types: Scenario
    ):
        VehicleTypeBlockAssignment().modify(db_session, {})
        db_session.flush()

        for rotation in db_session.query(Rotation).all():
            assert rotation.vehicle_type.name_short == "Diesel EN"

    def test_reassigns_only_listed_blocks(
        self, temp_db: Path, db_session: Session, scenario_with_diesel_types: Scenario
    ):
        rotations = db_session.query(Rotation).order_by(Rotation.id).all()
        target = rotations[0]

        VehicleTypeBlockAssignment().modify(
            db_session, {"VehicleTypeBlockAssignment.block_ids": [target.id]}
        )
        db_session.flush()

        assert target.vehicle_type.name_short == "Diesel EN"
        assert rotations[1].vehicle_type.name_short == "EN"

    def test_empty_block_list_is_a_no_op(
        self, temp_db: Path, db_session: Session, scenario_with_diesel_types: Scenario
    ):
        VehicleTypeBlockAssignment().modify(
            db_session, {"VehicleTypeBlockAssignment.block_ids": []}
        )
        db_session.flush()

        for rotation in db_session.query(Rotation).all():
            assert rotation.vehicle_type.name_short == "EN"

    def test_is_idempotent(
        self, temp_db: Path, db_session: Session, scenario_with_diesel_types: Scenario
    ):
        """Rotations already pointing at a diesel type are skipped, not re-prefixed."""
        step = VehicleTypeBlockAssignment()
        step.modify(db_session, {})
        db_session.flush()
        step.modify(db_session, {})
        db_session.flush()

        for rotation in db_session.query(Rotation).all():
            assert rotation.vehicle_type.name_short == "Diesel EN"

    def test_raises_when_no_diesel_types_exist(self, temp_db: Path, db_session: Session):
        scenario = Scenario(name="No Diesel", name_short="ND")
        db_session.add(scenario)
        db_session.flush()

        with pytest.raises(ValueError, match="No diesel vehicle types found"):
            VehicleTypeBlockAssignment().modify(db_session, {})

    def test_raises_when_a_rotation_has_no_diesel_counterpart(
        self, temp_db: Path, db_session: Session, scenario_with_diesel_types: Scenario
    ):
        orphan_type = VehicleType(
            scenario_id=scenario_with_diesel_types.id,
            name="Unmatched Bus",
            name_short="XX",
            battery_capacity=350.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=False,
            minimum_charging_power=10,
            charging_efficiency=0.95,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.2,
            energy_source=EnergySource.BATTERY_ELECTRIC,
        )
        db_session.add(orphan_type)
        db_session.flush()
        db_session.query(Rotation).order_by(Rotation.id).first().vehicle_type_id = orphan_type.id
        db_session.flush()

        with pytest.raises(ValueError, match="No diesel counterpart"):
            VehicleTypeBlockAssignment().modify(db_session, {})

    def test_document_params(self):
        params = VehicleTypeBlockAssignment.document_params()
        assert "VehicleTypeBlockAssignment.block_ids" in params


class TestCalibrateConsumptionLut:
    """Tests for the CalibrateConsumptionLut modifier."""

    @staticmethod
    def _write_measured_xlsx(path: Path, value: float = 2.0) -> None:
        """Write a small speed × temperature table in the consumption_lut_gn.xlsx layout."""
        import numpy as np
        import pandas as pd

        temperatures = [-10.0, 0.0, 10.0, 20.0, 30.0]
        speeds = [10.0, 20.0, 30.0, 40.0]
        data = np.full((len(speeds), len(temperatures)), value)
        data[0, 0] = np.nan  # ragged corner, like the real table
        df = pd.DataFrame(data, columns=temperatures)
        df.insert(0, "Temperatur (x) / Durchschnittsgeschwindigkeit (y)", speeds)
        df.to_excel(path, index=False)

    @pytest.fixture
    def scenario_with_vehicle_type(self, db_session: Session) -> Scenario:
        scenario = Scenario(name="Calibration Test", name_short="CAL")
        db_session.add(scenario)
        db_session.flush()
        vt = VehicleType(
            name="Solaris Urbino 18",
            name_short="GN",
            scenario_id=scenario.id,
            battery_capacity=600.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=19000.0,
            allowed_mass=19000.0 + 100 * 68,
            consumption=1.5,
        )
        db_session.add(vt)
        db_session.commit()
        return scenario

    @staticmethod
    def _lut_for(db_session: Session, name_short: str):
        from eflips.model import AssocVehicleTypeVehicleClass, ConsumptionLut, VehicleClass

        return (
            db_session.query(ConsumptionLut)
            .join(VehicleClass)
            .join(AssocVehicleTypeVehicleClass)
            .join(VehicleType)
            .filter(VehicleType.name_short == name_short)
            .one()
        )

    @staticmethod
    def _interpolate(lut, incline: float, t_amb: float, lol: float, speed: float) -> float:
        import numpy as np
        from scipy import interpolate

        pts = np.asarray(lut.data_points, dtype=float)
        vals = np.asarray(lut.values, dtype=float)
        scales = [np.unique(pts[:, i]) for i in range(4)]
        grid = np.full([len(s) for s in scales], np.nan)
        grid[tuple(np.searchsorted(scales[i], pts[:, i]) for i in range(4))] = vals
        f = interpolate.RegularGridInterpolator(tuple(scales), grid, bounds_error=False)
        return float(f([[incline, t_amb, lol, speed]])[0])

    def test_attaches_calibrated_lut(
        self, temp_db: Path, tmp_path: Path, db_session: Session, scenario_with_vehicle_type
    ):
        xlsx = tmp_path / "measured.xlsx"
        self._write_measured_xlsx(xlsx, value=2.0)

        CalibrateConsumptionLut(measured_lut_path=xlsx).modify(
            db_session, {"CalibrateConsumptionLut.vehicle_type_names": ["GN"]}
        )
        db_session.commit()

        vt = db_session.query(VehicleType).filter_by(name_short="GN").one()
        assert vt.consumption is None
        lut = self._lut_for(db_session, "GN")
        assert "measured.xlsx" in lut.name
        assert len(lut.values) == len(lut.data_points) > 0

        # At a measured point (flat, mean load) the table reproduces the measurement.
        for t_amb, speed in [(0.0, 20.0), (10.0, 30.0), (20.0, 40.0)]:
            assert self._interpolate(lut, 0.0, t_amb, 0.5, speed) == pytest.approx(2.0, rel=0.05)

        # The incline dependence of the regression model survives the calibration.
        downhill = self._interpolate(lut, -0.05, 10.0, 0.5, 30.0)
        flat = self._interpolate(lut, 0.0, 10.0, 0.5, 30.0)
        uphill = self._interpolate(lut, 0.05, 10.0, 0.5, 30.0)
        assert downhill < flat < uphill

    def test_slope_offset_is_not_scaled(self, temp_db: Path, db_session: Session):
        """Calibration scales the flat part only; the incline offset stays as generated."""
        import numpy as np
        from eflips.model import ConsumptionLut, VehicleClass

        scenario = Scenario(name="Slope", name_short="SL")
        db_session.add(scenario)
        db_session.flush()
        vt = VehicleType(
            name="Bus",
            name_short="B",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=19000.0,
            allowed_mass=19000.0 + 100 * 68,
            consumption=None,
        )
        db_session.add(vt)
        db_session.flush()
        vc = VehicleClass(scenario_id=scenario.id, name="vc", vehicle_types=[vt])
        db_session.add(vc)
        db_session.flush()
        lut = ConsumptionLut.from_vehicle_type(vt, vc)

        pts = np.asarray(lut.data_points, dtype=float)
        vals = np.asarray(lut.values, dtype=float)
        # Measured = twice the synthetic flat-ground values -> ratio field is exactly 2.
        on_plane = (pts[:, 0] == 0.0) & (pts[:, 2] == 0.5)
        measured = [
            (float(p[3]), float(p[1]), float(2.0 * v))
            for p, v in zip(pts[on_plane], vals[on_plane])
        ]
        scaled, stats = CalibrateConsumptionLut.calibrate_values(
            lut.data_points, lut.values, measured
        )
        scaled = np.asarray(scaled)
        assert stats["ratio_min"] == pytest.approx(2.0)
        assert stats["ratio_max"] == pytest.approx(2.0)

        # Flat part doubled ...
        assert np.allclose(scaled[on_plane], 2.0 * vals[on_plane])
        # ... while the incline offset relative to the flat slice is unchanged.
        flat_of = {}
        for p, v in zip(pts, vals):
            if p[0] == 0.0:
                flat_of[(p[1], p[2], p[3])] = v
        for p, v_old, v_new in zip(pts, vals, scaled):
            key = (p[1], p[2], p[3])
            assert v_new - 2.0 * flat_of[key] == pytest.approx(v_old - flat_of[key], abs=1e-9)

    def test_identity_when_measured_equals_model(self, temp_db: Path, db_session: Session):
        """Measured points taken from the synthetic table itself leave the values unchanged."""
        import numpy as np
        from eflips.model import ConsumptionLut, VehicleClass

        scenario = Scenario(name="Identity", name_short="ID")
        db_session.add(scenario)
        db_session.flush()
        vt = VehicleType(
            name="Bus",
            name_short="B",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=12000.0,
            allowed_mass=12000.0 + 70 * 68,
            consumption=None,
        )
        db_session.add(vt)
        db_session.flush()
        vc = VehicleClass(scenario_id=scenario.id, name="vc", vehicle_types=[vt])
        db_session.add(vc)
        db_session.flush()
        lut = ConsumptionLut.from_vehicle_type(vt, vc)

        pts = np.asarray(lut.data_points, dtype=float)
        vals = np.asarray(lut.values, dtype=float)
        on_plane = (pts[:, 0] == 0.0) & (pts[:, 2] == 0.5)
        measured = [
            (float(p[3]), float(p[1]), float(v)) for p, v in zip(pts[on_plane], vals[on_plane])
        ]

        scaled, stats = CalibrateConsumptionLut.calibrate_values(
            lut.data_points, lut.values, measured
        )
        assert stats["n_points"] == on_plane.sum()
        assert stats["ratio_min"] == pytest.approx(1.0)
        assert stats["ratio_max"] == pytest.approx(1.0)
        assert np.allclose(scaled, vals)

    def test_measured_points_outside_the_grid_are_ignored(
        self, temp_db: Path, db_session: Session
    ):
        """Measurements outside the synthetic grid are dropped instead of extrapolated."""
        import numpy as np
        from eflips.model import ConsumptionLut, VehicleClass

        scenario = Scenario(name="Outside", name_short="OU")
        db_session.add(scenario)
        db_session.flush()
        vt = VehicleType(
            name="Bus",
            name_short="B",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=19000.0,
            allowed_mass=19000.0 + 100 * 68,
            consumption=None,
        )
        db_session.add(vt)
        db_session.flush()
        vc = VehicleClass(scenario_id=scenario.id, name="vc", vehicle_types=[vt])
        db_session.add(vc)
        db_session.flush()
        lut = ConsumptionLut.from_vehicle_type(vt, vc)

        pts = np.asarray(lut.data_points, dtype=float)
        vals = np.asarray(lut.values, dtype=float)
        on_plane = (pts[:, 0] == 0.0) & (pts[:, 2] == 0.5)
        in_grid = [
            (float(p[3]), float(p[1]), float(2.0 * v))
            for p, v in zip(pts[on_plane], vals[on_plane])
        ]
        # A speed below and a temperature above the generated grid, with absurd values.
        min_speed, max_temp = pts[:, 3].min(), pts[:, 1].max()
        outside = [(min_speed - 1.0, 0.0, 99.0), (20.0, max_temp + 1.0, 99.0)]

        scaled, stats = CalibrateConsumptionLut.calibrate_values(
            lut.data_points, lut.values, in_grid + outside
        )
        assert stats["n_points"] == len(in_grid)
        assert stats["ratio_max"] == pytest.approx(2.0)
        assert np.allclose(np.asarray(scaled)[on_plane], 2.0 * vals[on_plane])

    def test_raises_without_masses(self, temp_db: Path, tmp_path: Path, db_session: Session):
        xlsx = tmp_path / "measured.xlsx"
        self._write_measured_xlsx(xlsx)
        scenario = Scenario(name="No mass", name_short="NM")
        db_session.add(scenario)
        db_session.flush()
        db_session.add(
            VehicleType(
                name="Bus",
                name_short="B",
                scenario_id=scenario.id,
                battery_capacity=400.0,
                battery_capacity_reserve=0.0,
                charging_curve=[[0, 150], [1, 150]],
                opportunity_charging_capable=True,
                minimum_charging_power=10,
                consumption=1.5,
            )
        )
        db_session.commit()

        with pytest.raises(ValueError, match="empty_mass and allowed_mass"):
            CalibrateConsumptionLut(measured_lut_path=xlsx).modify(db_session, {})

    def test_raises_for_unknown_vehicle_type(
        self, temp_db: Path, tmp_path: Path, db_session: Session, scenario_with_vehicle_type
    ):
        xlsx = tmp_path / "measured.xlsx"
        self._write_measured_xlsx(xlsx)
        with pytest.raises(ValueError, match="not found in scenario"):
            CalibrateConsumptionLut(measured_lut_path=xlsx).modify(
                db_session, {"CalibrateConsumptionLut.vehicle_type_names": ["XX"]}
            )

    def test_raises_when_multiple_scenarios(
        self, temp_db: Path, tmp_path: Path, db_session: Session
    ):
        xlsx = tmp_path / "measured.xlsx"
        self._write_measured_xlsx(xlsx)
        db_session.add(Scenario(name="S1", name_short="S1"))
        db_session.add(Scenario(name="S2", name_short="S2"))
        db_session.commit()
        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            CalibrateConsumptionLut(measured_lut_path=xlsx).modify(db_session, {})

    def test_document_params(self, tmp_path: Path):
        docs = CalibrateConsumptionLut(measured_lut_path=tmp_path / "x.xlsx").document_params()
        assert "CalibrateConsumptionLut.vehicle_type_names" in docs
