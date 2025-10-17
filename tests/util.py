import json
import math
import os
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz
import requests
from eflips.model import (
    Scenario,
    VehicleType,
    Rotation,
    Trip,
    TripType,
    Station,
    Route,
    Line,
    AssocRouteStation,
    Depot,
    Plan,
)
from geoalchemy2.shape import from_shape
from shapely import Point
from shapely.geometry import LineString
from sqlalchemy import func
from sqlalchemy.orm import Session

# Constants for geographic layout
CENTER_LAT = 52.520008
CENTER_LON = 13.404954
DEPOT_RING_DIAMETER = 20000  # meters

# Constants for network structure
NUM_DEPOTS = 2
LINES_PER_DEPOT = 6  # Must be even (2 lines per near terminus)
TRIPS_PER_LINE = 19
NEAR_TERMINUS_DISTANCE = 1000  # meters
FAR_TERMINUS_DISTANCE = 4000  # meters

# Constants for snapping
SNAP_RADIUS = 1000  # meters


def route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> tuple[float, any]:
    """
    Get route geometry and distance from OpenRouteService and cache results.

    Uses OpenRouteService Directions API to get the route between two points.
    Results are cached in data/cache/route_cache.json to avoid repeated API calls.

    Environment Variables:
        OPENROUTESERVICE_API_KEY: Required API key for ORS access
        OPENROUTESERVICE_BASE_URL: Optional custom ORS server URL (e.g., "http://localhost:8080/ors").
                     If not set, uses official API (https://api.openrouteservice.org)
                     with a warning about potential rate limits.

    Args:
        start_lat: Start latitude in WGS84
        start_lon: Start longitude in WGS84
        end_lat: End latitude in WGS84
        end_lon: End longitude in WGS84

    Returns:
        Tuple of (distance_meters, linestring_geom) where linestring_geom is a
        GeoAlchemy2 geometry object or None if routing failed
    """
    # Setup cache
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "route_cache.json"

    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = json.load(f)

    # Create cache key (rounded to 6 decimal places for consistency)
    cache_key = f"{start_lat:.6f},{start_lon:.6f}->{end_lat:.6f},{end_lon:.6f}"

    # Check cache
    if cache_key in cache:
        cached = cache[cache_key]
        distance = cached["distance"]
        coords = cached.get("geometry")
        if coords:
            # Convert list of [lon, lat] to LineString
            linestring = LineString([(lon, lat) for lon, lat in coords])
            geom = from_shape(linestring, srid=4326)
            return distance, geom
        return distance, None

    # Get API key from environment
    api_key = os.environ.get("OPENROUTESERVICE_API_KEY")
    if not api_key:
        # If no API key, return None for geometry (will fall back to straight line)
        warnings.warn(
            "No OpenRouteService API key provided. Make sure your server does not need an API key."
        )

    # Get base URL from environment, or use official API
    base_url = os.environ.get("OPENROUTESERVICE_BASE_URL")
    if base_url:
        # Use custom ORS server
        url = f"{base_url.rstrip('/')}/v2/directions/driving-car"
    else:
        # Use official API and warn about rate limits
        warnings.warn(
            "Using official OpenRouteService API. This may hit rate limits. "
            "Consider setting OPENROUTESERVICE_BASE_URL environment variable to use a custom ORS server.",
            UserWarning,
        )
        url = "https://api.openrouteservice.org/v2/directions/driving-car"

    params = {
        "api_key": api_key,
        "start": f"{start_lon},{start_lat}",
        "end": f"{end_lon},{end_lat}",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Extract result
    if data.get("features") and len(data["features"]) > 0:
        feature = data["features"][0]
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        # Get distance from properties
        distance = properties.get("summary", {}).get("distance", 0.0)

        # Get coordinates from geometry
        coords = geometry.get("coordinates", [])

        # Cache result
        cache[cache_key] = {
            "distance": distance,
            "geometry": coords,  # List of [lon, lat] pairs
        }

        # Save cache
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        # Convert to LineString geometry
        if coords:
            linestring = LineString([(lon, lat) for lon, lat in coords])
            geom = from_shape(linestring, srid=4326)
            return distance, geom
        return distance, None
    else:
        # No results, cache with no geometry
        cache[cache_key] = {"distance": None, "geometry": None}
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        return None, None


def snap_to_road(lat: float, lon: float) -> tuple[str | None, tuple[float, float]]:
    """
    Snap coordinates to the nearest road using OpenRouteService and cache results.

    Uses OpenRouteService Snapping API to find the nearest road to the given coordinates.
    Results are cached in data/cache/snap_cache.json to avoid repeated API calls.

    Environment Variables:
        OPENROUTESERVICE_API_KEY: Required API key for ORS access
        OPENROUTESERVICE_BASE_URL: Optional custom ORS server URL (e.g., "http://localhost:8080/ors").
                     If not set, uses official API (https://api.openrouteservice.org)
                     with a warning about potential rate limits.

    Args:
        lat: Latitude in WGS84
        lon: Longitude in WGS84

    Returns:
        Tuple of (street_name, (latitude, longitude)) where street_name may be None
        if no name was found, and coordinates are snapped to the road network
    """

    # Setup cache
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "snap_cache.json"

    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache = json.load(f)

    # Create cache key (rounded to 6 decimal places for consistency)
    cache_key = f"{lat:.6f},{lon:.6f}"

    # Check cache
    if cache_key in cache:
        cached = cache[cache_key]
        return cached.get("name"), tuple(cached["location"])

    # Get API key from environment, if it exists
    api_key = os.environ.get("OPENROUTESERVICE_API_KEY")
    if not api_key:
        # If no API key, return original coordinates with no name
        warnings.warn(
            "No OpenRouteService API key provided. Make sure your server does not need an API key."
        )

    # Get base URL from environment, or use official API
    base_url = os.environ.get("OPENROUTESERVICE_BASE_URL")
    if base_url:
        # Use custom ORS server
        url = f"{base_url.rstrip('/')}/v2/snap/driving-car"
    else:
        # Use official API and warn about rate limits
        warnings.warn(
            "Using official OpenRouteService API. This may hit rate limits. "
            "Consider setting OPENROUTESERVICE_BASE_URL environment variable to use a custom ORS server.",
            UserWarning,
        )
        url = "https://api.openrouteservice.org/v2/snap/driving-car"

    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    # Note: ORS expects [lon, lat] not [lat, lon]
    payload = {"locations": [[lon, lat]], "radius": SNAP_RADIUS}

    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Extract result
    if data.get("locations") and len(data["locations"]) > 0:
        result = data["locations"][0]
        snapped_location = result.get("location", [lon, lat])
        street_name = result.get("name")

        # Cache result (convert back to lat, lon order)
        cache[cache_key] = {
            "name": street_name,
            "location": [snapped_location[1], snapped_location[0]],  # [lat, lon]
        }

        # Save cache
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        return street_name, (snapped_location[1], snapped_location[0])
    else:
        # No results, cache the original coordinates
        cache[cache_key] = {"name": None, "location": [lat, lon]}
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        return None, (lat, lon)


def _create_station_at_projection(
    db_session: Session,
    scenario_id: int,
    base_geom: any,
    distance: float,
    angle: float,
    name: str,
    name_short: str,
    is_depot: bool = False,
) -> Station:
    """
    Create a station at a projected location from a base point.

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario
        base_geom: Base geometry to project from
        distance: Distance in meters
        angle: Angle in radians
        name: Station name (topological name, street name will be appended if found)
        name_short: Station short name
        is_depot: Whether this is a depot station

    Returns:
        Created Station object
    """
    # Project location from base
    geom_result = db_session.query(func.ST_Project(base_geom, distance, angle)).scalar()

    # Extract coordinates and snap to road
    point = db_session.query(func.ST_Y(geom_result), func.ST_X(geom_result)).one()
    street_name, (lat, lon) = snap_to_road(point[0], point[1])
    geom = from_shape(Point(lon, lat), srid=4326)

    # Append street name to station name if available
    full_name = name if street_name is None else f"{name} ({street_name})"

    # Create station
    if is_depot:
        station = Station(
            name=full_name,
            name_short=name_short,
            scenario_id=scenario_id,
            geom=geom,
            is_electrified=True,
            is_electrifiable=True,
            amount_charging_places=50,
            power_per_charger=150.0,
            power_total=7500.0,
            charge_type="depb",
            voltage_level="MV",
        )
    else:
        station = Station(
            name=full_name,
            name_short=name_short,
            scenario_id=scenario_id,
            geom=geom,
            is_electrified=False,
            is_electrifiable=True,
        )

    db_session.add(station)
    db_session.flush()
    return station


def _create_route_with_stations(
    db_session: Session,
    scenario_id: int,
    line_id: int,
    name: str,
    name_short: str,
    departure_station: Station,
    arrival_station: Station,
) -> Route:
    """
    Create a route with associated route-station relationships.

    Uses OpenRouteService routing to get actual road distance and geometry.
    Falls back to straight-line distance if routing fails.

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario
        line_id: ID of the line
        name: Route name
        name_short: Route short name
        departure_station: Departure station
        arrival_station: Arrival station

    Returns:
        Created Route object
    """
    # Extract station coordinates
    dep_point = db_session.query(
        func.ST_Y(departure_station.geom), func.ST_X(departure_station.geom)
    ).one()
    arr_point = db_session.query(
        func.ST_Y(arrival_station.geom), func.ST_X(arrival_station.geom)
    ).one()

    # Try to get routed geometry
    routed_distance, route_geom = route(dep_point[0], dep_point[1], arr_point[0], arr_point[1])

    # Calculate distance based on geometry
    if route_geom is not None:
        # Use ST_Length on the route geometry (returns meters for geography)
        distance = db_session.query(func.ST_Length(route_geom, True)).scalar()
    else:
        # Fall back to straight-line distance if routing failed
        distance = db_session.query(
            func.ST_Distance(departure_station.geom, arrival_station.geom, True)
        ).scalar()
        route_geom = None

    # Create route
    route_obj = Route(
        name=name,
        name_short=name_short,
        scenario_id=scenario_id,
        departure_station_id=departure_station.id,
        arrival_station_id=arrival_station.id,
        distance=distance,
        line_id=line_id,
        geom=route_geom,  # Will be None if routing failed
    )
    db_session.add(route_obj)
    db_session.flush()

    # Add AssocRouteStation entries
    db_session.add(
        AssocRouteStation(
            scenario_id=scenario_id,
            route_id=route_obj.id,
            station_id=departure_station.id,
            elapsed_distance=0.0,
        )
    )
    db_session.add(
        AssocRouteStation(
            scenario_id=scenario_id,
            route_id=route_obj.id,
            station_id=arrival_station.id,
            elapsed_distance=distance,
        )
    )

    return route_obj


def _create_trips_for_rotation(
    db_session: Session,
    scenario_id: int,
    rotation_id: int,
    route_depot_to_near: Route,
    route_near_to_far: Route,
    route_far_to_near: Route,
    route_near_to_depot: Route,
    route_far_to_depot: Route,
    trips_per_line: int,
    base_date: datetime,
) -> None:
    """
    Create trips for a rotation (depot -> service -> depot).

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario
        rotation_id: ID of the rotation
        route_depot_to_near: Route from depot to near terminus
        route_near_to_far: Route from near to far terminus
        route_far_to_near: Route from far to near terminus
        route_near_to_depot: Route from near terminus to depot
        route_far_to_depot: Route from far terminus to depot
        trips_per_line: Number of passenger trips
        base_date: Base datetime for the service day
    """
    start_time = base_date.replace(hour=5, minute=0, second=0, microsecond=0)
    current_time = start_time

    # First trip: Depot -> Near Terminus (EMPTY)
    trip_duration = timedelta(minutes=random.randint(20, 40))
    trip = Trip(
        scenario_id=scenario_id,
        route_id=route_depot_to_near.id,
        rotation_id=rotation_id,
        departure_time=current_time,
        arrival_time=current_time + trip_duration,
        trip_type=TripType.EMPTY,
    )
    db_session.add(trip)
    current_time += trip_duration

    # Middle trips: alternating between Near<->Far
    at_near_terminus = True  # We start at near terminus after first trip
    for trip_idx in range(trips_per_line):
        trip_duration = timedelta(minutes=random.randint(20, 40))

        if at_near_terminus:
            route = route_near_to_far
            at_near_terminus = False
        else:
            route = route_far_to_near
            at_near_terminus = True

        trip = Trip(
            scenario_id=scenario_id,
            route_id=route.id,
            rotation_id=rotation_id,
            departure_time=current_time,
            arrival_time=current_time + trip_duration,
            trip_type=TripType.PASSENGER,
        )
        db_session.add(trip)
        current_time += trip_duration

    # Last trip: return to depot (EMPTY)
    trip_duration = timedelta(minutes=random.randint(20, 40))

    if at_near_terminus:
        return_route = route_near_to_depot
    else:
        return_route = route_far_to_depot

    trip = Trip(
        scenario_id=scenario_id,
        route_id=return_route.id,
        rotation_id=rotation_id,
        departure_time=current_time,
        arrival_time=current_time + trip_duration,
        trip_type=TripType.EMPTY,
    )
    db_session.add(trip)


def _create_line_with_routes_and_trips(
    db_session: Session,
    scenario_id: int,
    vehicle_type_id: int,
    depot_station: Station,
    near_terminus: Station,
    far_terminus: Station,
    depot_idx: int,
    line_idx: int,
    trips_per_line: int,
    base_date: datetime,
) -> None:
    """
    Create a single line connecting depot <-> near terminus <-> far terminus.

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario
        vehicle_type_id: ID of the vehicle type
        depot_station: Depot station
        near_terminus: Near terminus station
        far_terminus: Far terminus station
        depot_idx: Index of the depot
        line_idx: Index of the line
        trips_per_line: Number of trips per line
        base_date: Base date for trips
    """
    line_name = f"D{depot_idx + 1}-L{line_idx + 1}"

    # Create line
    line = Line(
        name=line_name,
        name_short=line_name,
        scenario_id=scenario_id,
    )
    db_session.add(line)
    db_session.flush()

    # Create routes
    route_depot_to_near = _create_route_with_stations(
        db_session,
        scenario_id,
        line.id,
        f"{line_name} Depot to Near",
        f"{line_name}-DN",
        depot_station,
        near_terminus,
    )

    route_near_to_far = _create_route_with_stations(
        db_session,
        scenario_id,
        line.id,
        f"{line_name} Near to Far",
        f"{line_name}-NF",
        near_terminus,
        far_terminus,
    )

    route_far_to_near = _create_route_with_stations(
        db_session,
        scenario_id,
        line.id,
        f"{line_name} Far to Near",
        f"{line_name}-FN",
        far_terminus,
        near_terminus,
    )

    route_far_to_depot = _create_route_with_stations(
        db_session,
        scenario_id,
        line.id,
        f"{line_name} Far to Depot",
        f"{line_name}-FD",
        far_terminus,
        depot_station,
    )

    route_near_to_depot = _create_route_with_stations(
        db_session,
        scenario_id,
        line.id,
        f"{line_name} Near to Depot",
        f"{line_name}-ND",
        near_terminus,
        depot_station,
    )

    # Create rotation
    rotation = Rotation(
        name=f"{line_name} Rotation",
        scenario_id=scenario_id,
        vehicle_type_id=vehicle_type_id,
        allow_opportunity_charging=True,
    )
    db_session.add(rotation)
    db_session.flush()

    # Create trips
    _create_trips_for_rotation(
        db_session,
        scenario_id,
        rotation.id,
        route_depot_to_near,
        route_near_to_far,
        route_far_to_near,
        route_near_to_depot,
        route_far_to_depot,
        trips_per_line,
        base_date,
    )


def _create_depot_with_lines(
    db_session: Session,
    scenario_id: int,
    vehicle_type_id: int,
    depot_idx: int,
    num_depots: int,
    lines_per_depot: int,
    center_point: any,
    near_terminus_distance: float,
    far_terminus_distance: float,
    trips_per_line: int,
    base_date: datetime,
    depot_ring_diameter: float,
) -> None:
    """
    Create a depot with all its lines, routes, and trips.

    Network architecture: Creates lines_per_depot lines (must be even).
    Each line connects the depot to one near terminus and one far terminus.
    Near termini = lines_per_depot / 2 (distributed evenly around the depot)
    Far termini = lines_per_depot (twice as many as near termini)
    Lines = lines_per_depot (2 lines per near terminus)

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario
        vehicle_type_id: ID of the vehicle type
        depot_idx: Index of the depot
        num_depots: Total number of depots
        lines_per_depot: Number of lines per depot (must be even; determines near/far termini counts)
        center_point: Center point geometry for depot ring
        near_terminus_distance: Distance to near terminus
        far_terminus_distance: Distance to far terminus
        trips_per_line: Number of trips per line
        base_date: Base date for trips
        depot_ring_diameter: Diameter of the depot ring in meters

    Raises:
        ValueError: If lines_per_depot is odd
    """
    # Validate that lines_per_depot is even
    if lines_per_depot % 2 != 0:
        raise ValueError(f"lines_per_depot must be even, got {lines_per_depot}")

    # Calculate angle for this depot
    angle_radians = (2 * math.pi * depot_idx) / num_depots

    # Create depot station
    depot_station = _create_station_at_projection(
        db_session,
        scenario_id,
        center_point,
        depot_ring_diameter / 2,
        angle_radians,
        f"Depot {depot_idx + 1}",
        f"D{depot_idx + 1}",
        is_depot=True,
    )

    # Create default plan for depot
    plan = Plan(
        name=f"Depot {depot_idx + 1} Default Plan",
        scenario_id=scenario_id,
    )
    db_session.add(plan)
    db_session.flush()

    # Create depot
    depot = Depot(
        name=f"Depot {depot_idx + 1}",
        name_short=f"D{depot_idx + 1}",
        scenario_id=scenario_id,
        station_id=depot_station.id,
        default_plan_id=plan.id,
    )
    db_session.add(depot)
    db_session.flush()

    # Calculate number of near and far termini
    num_near_termini = lines_per_depot // 2  # Half the number of lines
    num_far_termini = lines_per_depot  # Same as number of lines

    # Create near termini - distributed evenly around the depot
    near_termini = []
    for near_idx in range(num_near_termini):
        terminus_angle = (2 * math.pi * near_idx) / num_near_termini
        near_terminus = _create_station_at_projection(
            db_session,
            scenario_id,
            depot_station.geom,
            near_terminus_distance,
            terminus_angle,
            f"D{depot_idx + 1}-N{near_idx + 1}",
            f"D{depot_idx + 1}-N{near_idx + 1}",
            is_depot=False,
        )
        near_termini.append(near_terminus)

    # Create far termini - distributed at twice the density (2 per near terminus)
    far_termini = []
    for far_idx in range(num_far_termini):
        terminus_angle = (2 * math.pi * far_idx) / num_far_termini
        far_terminus = _create_station_at_projection(
            db_session,
            scenario_id,
            depot_station.geom,
            far_terminus_distance,
            terminus_angle,
            f"D{depot_idx + 1}-F{far_idx + 1}",
            f"D{depot_idx + 1}-F{far_idx + 1}",
            is_depot=False,
        )
        far_termini.append(far_terminus)

    # Create lines: each line connects one near terminus to one far terminus
    # We create lines_per_depot lines total (2 lines per near terminus)
    for line_idx in range(lines_per_depot):
        # Determine which near terminus this line uses (each near gets 2 lines)
        near_idx = line_idx // 2
        near_terminus = near_termini[near_idx]

        # Each near terminus connects to 2 consecutive far termini
        far_terminus = far_termini[line_idx]

        _create_line_with_routes_and_trips(
            db_session,
            scenario_id,
            vehicle_type_id,
            depot_station,
            near_terminus,
            far_terminus,
            depot_idx,
            line_idx,
            trips_per_line,
            base_date,
        )


@pytest.fixture
def multi_depot_scenario(
    db_session: Session,
    num_depots: int = NUM_DEPOTS,
    lines_per_depot: int = LINES_PER_DEPOT,
    trips_per_line: int = TRIPS_PER_LINE,
    near_terminus_distance: float = NEAR_TERMINUS_DISTANCE,
    far_terminus_distance: float = FAR_TERMINUS_DISTANCE,
    depot_ring_diameter: float = DEPOT_RING_DIAMETER,
) -> Scenario:
    """
    Create a scenario with multiple depots and randomized bus schedules.

    Network architecture:
    - Each depot has lines_per_depot / 2 near termini distributed evenly in a ring
    - Each depot has lines_per_depot far termini (twice as many as near termini)
    - Each depot has lines_per_depot lines total (2 lines per near terminus)
    - Each line connects depot -> near terminus -> far terminus and back

    Args:
        db_session: SQLAlchemy session
        num_depots: Number of depots (default: 2)
        lines_per_depot: Number of lines per depot (must be even; default: 6)
        trips_per_line: Number of passenger trips per line per day (default: 19)
        near_terminus_distance: Distance from depot to near terminus in meters (default: 1000)
        far_terminus_distance: Distance from depot to far terminus in meters (default: 3000)
        depot_ring_diameter: Diameter of the depot ring in meters (default: 20000)

    Returns:
        Scenario with depots, lines, routes, and trips

    Raises:
        ValueError: If lines_per_depot is odd
    """
    # Set random seed for reproducibility in tests
    random.seed(42)

    # Create scenario
    scenario = Scenario(name="Multi-Depot Test Scenario", name_short="MDTS")
    db_session.add(scenario)
    db_session.flush()

    # Create a single vehicle type for all depots
    vehicle_type = VehicleType(
        name="Electric Bus 12m",
        name_short="EB12",
        scenario_id=scenario.id,
        battery_capacity=350.0,
        battery_capacity_reserve=0.0,
        charging_curve=[[0, 150], [1, 150]],
        opportunity_charging_capable=True,
        minimum_charging_power=10,
        empty_mass=10000,
        allowed_mass=20000,
        consumption=1.2,
    )
    db_session.add(vehicle_type)
    db_session.flush()

    # Berlin timezone
    berlin_tz = pytz.timezone("Europe/Berlin")
    # Start date: arbitrary date for testing
    base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

    # Center point for depot ring
    center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

    # Create depots arranged in a circle
    for depot_idx in range(num_depots):
        _create_depot_with_lines(
            db_session,
            scenario.id,
            vehicle_type.id,
            depot_idx,
            num_depots,
            lines_per_depot,
            center_point,
            near_terminus_distance,
            far_terminus_distance,
            trips_per_line,
            base_date,
            depot_ring_diameter,
        )

    db_session.commit()
    return scenario


def generate_network_map(db_session: Session, scenario_id: int, output_file: str) -> None:
    """
    Generate an interactive Folium map visualization of the bus network.

    Args:
        db_session: SQLAlchemy session
        scenario_id: ID of the scenario to visualize
        output_file: Path to save the HTML map file
    """
    import folium
    from geoalchemy2.shape import to_shape

    # Create map centered on Berlin
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=12)

    # Create feature groups for layers
    depot_layer = folium.FeatureGroup(name="Depots", show=True)
    terminus_layer = folium.FeatureGroup(name="Termini", show=True)
    passenger_routes_layer = folium.FeatureGroup(name="Passenger Routes", show=True)
    empty_routes_layer = folium.FeatureGroup(name="Empty Routes", show=True)

    # Query all stations and depots
    all_stations = db_session.query(Station).filter_by(scenario_id=scenario_id).all()
    all_depots = db_session.query(Depot).filter_by(scenario_id=scenario_id).all()

    # Get depot station IDs
    depot_station_ids = {depot.station_id for depot in all_depots}

    # Separate depots and termini
    depot_stations = [s for s in all_stations if s.id in depot_station_ids]
    terminus_stations = [s for s in all_stations if s.id not in depot_station_ids]

    print(f"Plotting {len(depot_stations)} depots and {len(terminus_stations)} termini")

    # Plot depot stations with special marker
    for station in depot_stations:
        point = to_shape(station.geom)
        folium.Marker(
            location=[point.y, point.x],
            popup=f"<b>{station.name}</b><br>Depot<br>{station.amount_charging_places} charging places",
            tooltip=station.name_short,
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(depot_layer)

    # Plot terminus stations
    for station in terminus_stations:
        point = to_shape(station.geom)
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=6,
            popup=f"<b>{station.name}</b><br>Terminus",
            tooltip=station.name_short,
            color="blue",
            fill=True,
            fillColor="lightblue",
            fillOpacity=0.7,
        ).add_to(terminus_layer)

    # Query all routes and their trips
    all_routes = db_session.query(Route).filter_by(scenario_id=scenario_id).all()

    print(f"Plotting {len(all_routes)} routes")

    # Analyze each route to determine if it has only empty trips or passenger trips
    for route in all_routes:
        trips = db_session.query(Trip).filter_by(route_id=route.id).all()

        if not trips:
            continue

        # Check trip types
        has_passenger_trips = any(t.trip_type == TripType.PASSENGER for t in trips)
        has_only_empty_trips = all(t.trip_type == TripType.EMPTY for t in trips)

        # Determine color based on trip types
        if has_only_empty_trips:
            color = "gray"
            popup_text = f"<b>{route.name_short}</b><br>Empty trips only<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = empty_routes_layer
        elif has_passenger_trips:
            color = "green"
            popup_text = f"<b>{route.name_short}</b><br>Passenger service<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = passenger_routes_layer
        else:
            color = "orange"
            popup_text = f"<b>{route.name_short}</b><br>Mixed service<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = passenger_routes_layer

        # Draw route geometry if available, otherwise draw straight line
        if route.geom is not None:
            # Convert route geometry to list of coordinates
            route_shape = to_shape(route.geom)
            if hasattr(route_shape, "coords"):
                # LineString geometry
                locations = [[lat, lon] for lon, lat in route_shape.coords]
                folium.PolyLine(
                    locations=locations,
                    popup=popup_text,
                    color=color,
                    weight=3,
                    opacity=0.7,
                ).add_to(layer)
        else:
            # Fall back to straight line between stations
            dep_station = (
                db_session.query(Station).filter_by(id=route.departure_station_id).first()
            )
            arr_station = db_session.query(Station).filter_by(id=route.arrival_station_id).first()

            dep_point = to_shape(dep_station.geom)
            arr_point = to_shape(arr_station.geom)

            folium.PolyLine(
                locations=[[dep_point.y, dep_point.x], [arr_point.y, arr_point.x]],
                popup=popup_text,
                color=color,
                weight=3,
                opacity=0.7,
            ).add_to(layer)

    # Add all layers to map
    depot_layer.add_to(m)
    terminus_layer.add_to(m)
    passenger_routes_layer.add_to(m)
    empty_routes_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 200px; height: 160px;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <h4 style="margin-top:0">Legend</h4>
    <p><i class="fa fa-home" style="color:red"></i> Depot</p>
    <p><i class="fa fa-circle" style="color:lightblue"></i> Terminus</p>
    <p><span style="color:green; font-weight:bold">━━</span> Passenger service</p>
    <p><span style="color:gray; font-weight:bold">━━</span> Empty trips only</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_file)
    print(f"\nMap saved to {output_file}")


if __name__ == "__main__":
    import folium
    from eflips.model import create_engine
    from sqlalchemy.orm import sessionmaker
    from eflips.model import setup_database
    from geoalchemy2.shape import to_shape

    # Create in-memory database
    engine = create_engine("sqlite:///:memory:")
    setup_database(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Create random bus network
    print("Creating random bus network...")
    random.seed(42)

    # Create scenario
    scenario = Scenario(name="Visualization Test", name_short="VIS")
    session.add(scenario)
    session.flush()

    # Create vehicle type
    vehicle_type = VehicleType(
        name="Electric Bus 12m",
        name_short="EB12",
        scenario_id=scenario.id,
        battery_capacity=350.0,
        battery_capacity_reserve=0.0,
        charging_curve=[[0, 150], [1, 150]],
        opportunity_charging_capable=True,
        minimum_charging_power=10,
        empty_mass=10000,
        allowed_mass=20000,
        consumption=1.2,
    )
    session.add(vehicle_type)
    session.flush()

    # Berlin timezone
    berlin_tz = pytz.timezone("Europe/Berlin")
    base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

    # Center point for depot ring
    center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

    # Create network with 6 depots, 10 lines per depot (5 near termini, 10 far termini per depot)
    num_depots = 3
    lines_per_depot = 10  # Must be even
    trips_per_line = 9

    for depot_idx in range(num_depots):
        _create_depot_with_lines(
            session,
            scenario.id,
            vehicle_type.id,
            depot_idx,
            num_depots,
            lines_per_depot,
            center_point,
            NEAR_TERMINUS_DISTANCE,  # near_terminus_distance
            FAR_TERMINUS_DISTANCE,  # far_terminus_distance
            trips_per_line,
            base_date,
            DEPOT_RING_DIAMETER,
        )

    session.commit()
    print(f"Created {num_depots} depots with {lines_per_depot} lines each")

    # Create map centered on Berlin
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=12)

    # Create feature groups for layers
    depot_layer = folium.FeatureGroup(name="Depots", show=True)
    terminus_layer = folium.FeatureGroup(name="Termini", show=True)
    passenger_routes_layer = folium.FeatureGroup(name="Passenger Routes", show=True)
    empty_routes_layer = folium.FeatureGroup(name="Empty Routes", show=True)

    # Query all stations and depots
    all_stations = session.query(Station).filter_by(scenario_id=scenario.id).all()
    all_depots = session.query(Depot).filter_by(scenario_id=scenario.id).all()

    # Get depot station IDs
    depot_station_ids = {depot.station_id for depot in all_depots}

    # Separate depots and termini
    depot_stations = [s for s in all_stations if s.id in depot_station_ids]
    terminus_stations = [s for s in all_stations if s.id not in depot_station_ids]

    print(f"Plotting {len(depot_stations)} depots and {len(terminus_stations)} termini")

    # Plot depot stations with special marker
    for station in depot_stations:
        point = to_shape(station.geom)
        folium.Marker(
            location=[point.y, point.x],
            popup=f"<b>{station.name}</b><br>Depot<br>{station.amount_charging_places} charging places",
            tooltip=station.name_short,
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(depot_layer)

    # Plot terminus stations
    for station in terminus_stations:
        point = to_shape(station.geom)
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=6,
            popup=f"<b>{station.name}</b><br>Terminus",
            tooltip=station.name_short,
            color="blue",
            fill=True,
            fillColor="lightblue",
            fillOpacity=0.7,
        ).add_to(terminus_layer)

    # Query all routes and their trips
    all_routes = session.query(Route).filter_by(scenario_id=scenario.id).all()

    print(f"Plotting {len(all_routes)} routes")

    # Analyze each route to determine if it has only empty trips or passenger trips
    for route in all_routes:
        trips = session.query(Trip).filter_by(route_id=route.id).all()

        if not trips:
            continue

        # Check trip types
        has_passenger_trips = any(t.trip_type == TripType.PASSENGER for t in trips)
        has_only_empty_trips = all(t.trip_type == TripType.EMPTY for t in trips)

        # Determine color based on trip types
        if has_only_empty_trips:
            color = "gray"
            popup_text = f"<b>{route.name_short}</b><br>Empty trips only<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = empty_routes_layer
        elif has_passenger_trips:
            color = "green"
            popup_text = f"<b>{route.name_short}</b><br>Passenger service<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = passenger_routes_layer
        else:
            color = "orange"
            popup_text = f"<b>{route.name_short}</b><br>Mixed service<br>{len(trips)} trips<br>Distance: {route.distance:.0f}m"
            layer = passenger_routes_layer

        # Draw route geometry if available, otherwise draw straight line
        if route.geom is not None:
            # Convert route geometry to list of coordinates
            route_shape = to_shape(route.geom)
            if hasattr(route_shape, "coords"):
                # LineString geometry
                locations = [[lat, lon] for lon, lat in route_shape.coords]
                folium.PolyLine(
                    locations=locations,
                    popup=popup_text,
                    color=color,
                    weight=3,
                    opacity=0.7,
                ).add_to(layer)
        else:
            # Fall back to straight line between stations
            dep_station = session.query(Station).filter_by(id=route.departure_station_id).first()
            arr_station = session.query(Station).filter_by(id=route.arrival_station_id).first()

            dep_point = to_shape(dep_station.geom)
            arr_point = to_shape(arr_station.geom)

            folium.PolyLine(
                locations=[[dep_point.y, dep_point.x], [arr_point.y, arr_point.x]],
                popup=popup_text,
                color=color,
                weight=3,
                opacity=0.7,
            ).add_to(layer)

    # Add all layers to map
    depot_layer.add_to(m)
    terminus_layer.add_to(m)
    passenger_routes_layer.add_to(m)
    empty_routes_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 200px; height: 160px;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <h4 style="margin-top:0">Legend</h4>
    <p><i class="fa fa-home" style="color:red"></i> Depot</p>
    <p><i class="fa fa-circle" style="color:lightblue"></i> Terminus</p>
    <p><span style="color:green; font-weight:bold">━━</span> Passenger service</p>
    <p><span style="color:gray; font-weight:bold">━━</span> Empty trips only</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    output_file = "bus_network_visualization.html"
    m.save(output_file)
    print(f"\nMap saved to {output_file}")
    print("Open this file in a web browser to view the visualization")

    session.close()
