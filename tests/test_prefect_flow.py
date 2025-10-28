"""Tests for Prefect flow integration with eflips-x pipeline steps."""

import tempfile
from pathlib import Path
from typing import List

import pytest
from prefect import flow

from eflips.x.framework import PipelineContext
from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.simulation import DepotGenerator
from tests.steps.analyzers.test_dummy_analyzer import TripDistanceAnalyzer


class TestPrefectFlow:
    """Test suite for Prefect flow integration."""

    @pytest.fixture
    def work_dir(self) -> Path:
        """Create a temporary work directory for pipeline execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def xml_files(self, test_data_dir: Path) -> List[Path]:
        """Get list of XML files from test data directory."""
        xml_files = sorted(test_data_dir.glob("*.xml"))
        assert len(xml_files) > 0, f"No XML files found in {test_data_dir}"
        # Use only first 2 files for faster testing
        return xml_files[:2]

    @pytest.fixture
    def pipeline_params(self) -> dict:
        """Get default pipeline parameters."""
        return {
            "log_level": "WARNING",
            "BVGXMLIngester.multithreading": False,
        }

    def test_simple_flow_generator_only(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test a simple flow with just the generator step."""

        @flow(name="test-generator-flow")
        def test_flow():
            # Create pipeline context
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Create and execute generator
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            result_db = ingester.execute(context)

            return result_db

        # Run the flow
        test_flow()

    def test_flow_generator_and_analyzer(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test a flow with generator followed by analyzer."""

        @flow(name="test-generator-analyzer-flow")
        def test_flow():
            # Create pipeline context
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Step 1: Generate database
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            ingester.execute(context)

            # Step 2: Analyze database
            analyzer = TripDistanceAnalyzer(cache_enabled=False)
            total_distance = analyzer.execute(context)

            return total_distance

        # Run the flow
        total_distance = test_flow()

        # Verify results
        assert isinstance(total_distance, float)
        assert total_distance > 0, "Total distance should be positive"

    def test_flow_with_context_chaining(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test that context properly chains database state between steps."""

        @flow(name="test-context-chaining-flow")
        def test_flow():
            # Create pipeline context
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Initially no current_db
            assert context.current_db is None

            # Step 1: Generate database
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            ingester.execute(context)

            # After generator, current_db should be set
            assert context.current_db is not None
            db_path = context.current_db

            # Step 2: Analyze database
            analyzer = TripDistanceAnalyzer(cache_enabled=False)
            total_distance = analyzer.execute(context)

            # current_db should still be set
            assert context.current_db == db_path

            return context, total_distance

        # Run the flow
        context, total_distance = test_flow()

        # Verify context state
        assert context.step_count == 1  # Generator counts as 1 step
        assert context.current_db is not None
        assert context.current_db.exists()

    def test_flow_with_artifacts(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test that analyzer results are stored in context artifacts."""

        @flow(name="test-artifacts-flow")
        def test_flow():
            # Create pipeline context
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Step 1: Generate database
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            ingester.execute(context)

            # Step 2: Analyze database
            analyzer = TripDistanceAnalyzer(cache_enabled=False)
            total_distance = analyzer.execute(context)

            return context, total_distance

        # Run the flow
        context, total_distance = test_flow()

        # Verify artifacts
        assert "TripDistanceAnalyzer" in context.artifacts
        assert context.artifacts["TripDistanceAnalyzer"] == total_distance

    def test_flow_step_count_increments(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test that step_count increments correctly."""

        @flow(name="test-step-count-flow")
        def test_flow():
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            assert context.step_count == 0

            # Step 1: Generator
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            ingester.execute(context)

            assert context.step_count == 1

            return context

        # Run the flow
        context = test_flow()
        assert context.step_count == 1

    def test_multiple_analyzers_in_flow(
        self, work_dir: Path, xml_files: List[Path], pipeline_params: dict
    ):
        """Test running multiple analyzers in sequence."""

        @flow(name="test-multiple-analyzers-flow")
        def test_flow():
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Generate database
            ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
            ingester.execute(context)

            # Run first analyzer
            analyzer1 = TripDistanceAnalyzer(cache_enabled=False)
            distance1 = analyzer1.execute(context)

            # Run second analyzer (same type, different instance)
            analyzer2 = TripDistanceAnalyzer(cache_enabled=False)
            distance2 = analyzer2.execute(context)

            return distance1, distance2

        # Run the flow
        distance1, distance2 = test_flow()

        # Both analyzers should return the same result
        assert distance1 == distance2
        assert distance1 > 0

    def test_full_simulation_flow(self, work_dir: Path, pipeline_params: dict, db_session):
        """Test a complete flow: Vehicle Scheduling -> Depot Assignment -> Simulation."""
        from tests.util import multi_depot_scenario
        from eflips.x.steps.modifiers.scheduling import VehicleScheduling, DepotAssignment
        from eflips.x.steps.modifiers.simulation import Simulation
        from eflips.model import ChargeType, Depot, VehicleType, Event

        @flow(name="test-full-simulation-flow")
        def test_flow():
            # Create pipeline context
            context = PipelineContext(work_dir=work_dir, params=pipeline_params)

            # Create a multi-depot scenario using the fixture function
            # Use higher trips_per_line to force rotation splitting
            scenario = multi_depot_scenario(
                db_session,
                num_depots=2,
                lines_per_depot=4,
                trips_per_line=30,  # High number to force rotation splitting
            )

            # Step 1: Run Vehicle Scheduling in DEPOT mode
            vehicle_scheduler = VehicleScheduling()
            vs_params = {
                "VehicleScheduling.charge_type": ChargeType.DEPOT,
                "VehicleScheduling.battery_margin": 0.1,
            }
            vehicle_scheduler.modify(session=db_session, params=vs_params)

            # Step 2: Run Depot Assignment
            # Get all depots for configuration
            all_depots = db_session.query(Depot).filter_by(scenario_id=scenario.id).all()
            all_vehicle_types = (
                db_session.query(VehicleType).filter_by(scenario_id=scenario.id).all()
            )

            depot_config = []
            for depot in all_depots:
                depot_config.append(
                    {
                        "depot_station": depot.station_id,
                        "capacity": 100,  # Large capacity to accept all rotations
                        "vehicle_type": [vt.id for vt in all_vehicle_types],
                        "name": depot.name,
                    }
                )

            depot_assigner = DepotAssignment()
            da_params = {
                "DepotAssignment.depot_config": depot_config,
                "DepotAssignment.depot_usage": 0.9,
                "DepotAssignment.step_size": 0.2,
                "DepotAssignment.max_iterations": 1,
            }
            depot_assigner.modify(session=db_session, params=da_params)

            # Step 3: Add Depots
            depot_creator = DepotGenerator()
            depot_creator.modify(session=db_session, params={})

            # Step 4: Run Simulation
            simulator = Simulation()
            sim_params = {}
            simulator.modify(session=db_session, params=sim_params)

            db_session.commit()

            return scenario

        # Run the flow
        scenario = test_flow()

        # Verify that simulation events were created
        events_count = db_session.query(Event).filter(Event.scenario_id == scenario.id).count()
        assert events_count > 0, "Should have created simulation events"
