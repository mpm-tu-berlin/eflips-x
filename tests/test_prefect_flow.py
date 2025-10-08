"""Tests for Prefect flow integration with eflips-x pipeline steps."""

import tempfile
from pathlib import Path
from typing import List

import pytest
from prefect import flow

from eflips.x.framework import PipelineContext
from eflips.x.steps.generators import BVGXMLIngester
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
