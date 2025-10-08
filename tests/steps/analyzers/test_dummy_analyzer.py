"""Dummy analyzer for testing pipeline flows."""

from pathlib import Path
from typing import Dict, Any

import eflips.model
from eflips.model import Trip, Route
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer


class TripDistanceAnalyzer(Analyzer):
    """
    Dummy analyzer that calculates the total distance of all passenger trips.
    Used for testing Prefect flow integration.
    """

    def __init__(self, code_version: str = "v1", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def analyze(self, db: Path, params: Dict[str, Any]) -> float:
        """
        Analyze the database and return the total distance of all trips.

        Args:
            db: Path to the database file
            params: Analysis parameters (unused in this dummy analyzer)

        Returns:
            Total distance of all trips in meters
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Query all trips and sum their route distances
            total_distance = (
                session.query(Route.distance).join(Trip).filter(Route.distance.isnot(None)).all()
            )

            # Sum up all distances
            result = sum(distance[0] for distance in total_distance if distance[0] is not None)

            return float(result)
        finally:
            session.close()
            engine.dispose()
