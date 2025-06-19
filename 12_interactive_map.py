#! /usr/bin/env python3


import os
from datetime import datetime, timedelta
from typing import Dict, List

import eflips.eval.output.prepare
import eflips.eval.output.visualize
import folium
import folium.plugins
import geoalchemy2
import numpy as np
import pytz
import sqlalchemy.orm.session
from eflips.model import (
    Event,
    EventType,
    Route,
    Trip,
    Scenario,
    Station,
    Depot,
    Vehicle,
    Area,
    ChargeType,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload
from tqdm.auto import tqdm

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)

OUTPUT_FOLDER = "output_interactive"


def icon_for_number(number: int) -> folium.Icon:
    """
    Get an icon for a given number.
    :param number: A number.
    :return: An icon.
    """
    return folium.Icon(color="black", icon=str(int(number)), prefix="fa")


def single_charging_station_location_and_utilization(
    scenario: Scenario, session: Session, station_id: int
) -> folium.Marker:
    """
    Get information about a single charging station in a given scenario.

    This returns a dictionary with the following keys:
    - station_id: The ID of the station.
    - station_name: The name of the station.
    - latitude: The latitude of the station.
    - longitude: The longitude of the station.
    - number_of_charging_places: The number of charging places at the station.
    - utilization: The utilization of the station in the scenario (in percent).
    - link_to_graph: A link to a graph showing the utilization of the station.

    :param scenario:
    :param session:
    :param station_id:
    :return:
    """
    station = session.query(Station).filter(Station.id == station_id).one()
    station_name = station.name
    # Decode the location using geoalchemy2 and shapely
    point = geoalchemy2.shape.to_shape(station.geom)
    latitude = point.y
    longitude = point.x

    # Use the power_and_occupancy function to get the power and occupancy data
    df = eflips.eval.output.prepare.power_and_occupancy(
        area_id=None,
        session=session,
        station_id=station_id,
        temporal_resolution=60,
        sim_start_time=START_OF_SIMULATION,
        sim_end_time=END_OF_SIMULATION,
    )

    number_of_charging_places = max(df["occupancy_total"])
    utilization = 100 * df["occupancy_total"].mean() / number_of_charging_places

    # Save the visualization in a file
    # The folder will be the scenario name + station_power_and_occupancy
    # The filename will be station_id
    folder_name = os.path.join(
        OUTPUT_FOLDER, scenario.name_short, "station_power_and_occupancy"
    )
    folder_name_from_inside = os.path.join(
        scenario.name_short, "station_power_and_occupancy"
    )
    os.makedirs(folder_name, exist_ok=True)

    fig = eflips.eval.output.visualize.power_and_occupancy(df)
    fig.update_layout(title_text=f"{station_name}")
    fig.write_html(os.path.join(folder_name, f"{station_id}.html"))
    link_to_graph = f"<a href='{os.path.join(folder_name_from_inside, f'{station_id}.html')}'>Zeitreihe</a>"

    list_of_lines = set()

    # Get the opportunity charging events at the station
    opportunity_charging_events = (
        session.query(Event)
        .filter(Event.station_id == station_id)
        .filter(Event.event_type == EventType.CHARGING_OPPORTUNITY)
        .all()
    )
    for event in opportunity_charging_events:
        # Get the preceding driving event's route
        driving_event = (
            session.query(Event)
            .filter(Event.vehicle_id == event.vehicle_id)
            .filter(Event.time_end <= event.time_start)
            .filter(Event.event_type == EventType.DRIVING)
            .order_by(Event.time_end.desc())
            .options(
                joinedload(Event.trip).joinedload(Trip.route).joinedload(Route.line)
            )
            .first()
        )
        if driving_event is not None:
            list_of_lines.add(driving_event.trip.route.line.name)

    list_of_lines = sorted(list_of_lines)

    popup_text = f"""
    <h3>{station_name}</h3>
    <p>Anzahl Ladeplätze: {number_of_charging_places}</p>
    <p>Ausnutzung: {utilization:.2f}%</p>
    <p>{link_to_graph}</p>
    <p>Linien: {", ".join(list_of_lines)}</p>
    """

    marker = folium.Marker(
        location=[latitude, longitude],
        popup=popup_text,
        tooltip=station_name,
        icon=icon_for_number(number_of_charging_places),
    )
    return marker


def scenario_charging_station_location_and_utilization(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> List[folium.Marker]:
    """
    Get information about the charging stations in a given scenario.

    :param scenario:
    :param session:
    :return:
    """
    stations = (
        session.query(Station)
        .filter(Station.scenario_id == scenario.id)
        .filter(Station.is_electrified == True)
        .filter(Station.charge_type == ChargeType.OPPORTUNITY)
        .all()
    )
    result = []
    for station in tqdm(stations, desc="Stations"):
        result.append(
            single_charging_station_location_and_utilization(
                scenario, session, station.id
            )
        )

    return result


def single_depot_location_and_utilization(scenario, session, depot_id):
    """
    Get information about a single depot in a given scenario.

    :param scenario:
    :param session:
    :param depot_id:
    :return:
    """
    depot_station = (
        session.query(Station).join(Depot).filter(Depot.id == depot_id).one()
    )
    name = depot_station.name
    point = geoalchemy2.shape.to_shape(depot_station.geom)
    latitude = point.y
    longitude = point.x

    folder_name = os.path.join(
        OUTPUT_FOLDER, scenario.name_short, "depot_power_and_occupancy"
    )
    folder_name_from_inside = os.path.join(
        scenario.name_short, "depot_power_and_occupancy"
    )
    os.makedirs(folder_name, exist_ok=True)

    # Get the power and occupancy data
    all_areas = session.query(Area).filter(Area.depot_id == depot_id).all()
    area_ids = [area.id for area in all_areas]
    df = eflips.eval.output.prepare.power_and_occupancy(
        area_id=area_ids,
        session=session,
        station_id=None,
        sim_start_time=START_OF_SIMULATION,
        sim_end_time=END_OF_SIMULATION,
    )
    # Save the visualization in a file
    fig = eflips.eval.output.visualize.power_and_occupancy(df)
    fig.update_layout(title_text=f"{name}")
    fig.write_html(os.path.join(folder_name, f"{depot_id}_power.html"))
    link_to_power_and_occupancy = f"<a href='{os.path.join(folder_name_from_inside, f'{depot_id}_power.html')}'>Fahrzeuganzahl und Leistung</a>"

    # Visualize a timeline for what happens in the depot
    vehicles = (
        session.query(Vehicle)
        .join(Event)
        .join(Area)
        .filter(Area.depot_id == depot_id)
        .all()
    )
    vehicle_ids = [vehicle.id for vehicle in vehicles]
    df = eflips.eval.output.prepare.depot_event(scenario.id, session, vehicle_ids)
    color_scheme = "event_type"
    fig = eflips.eval.output.visualize.depot_event(df, color_scheme=color_scheme)
    # Set the x-axis to the simulation time
    fig.update_layout(xaxis_range=[START_OF_SIMULATION, END_OF_SIMULATION])
    fig.update_layout(title_text=f"{name}")

    fig.write_html(os.path.join(folder_name, f"{depot_id}_timeline.html"))
    link_to_timeline = f"<a href='{os.path.join(folder_name_from_inside, f'{depot_id}_timeline.html')}'>Zeitreihe</a>"

    popup_text = f"""
    <h3>{name}</h3>
    <p>{link_to_power_and_occupancy}</p>
    <p>{link_to_timeline}</p>
    """

    return folium.Marker(
        location=[latitude, longitude],
        popup=popup_text,
        tooltip=name,
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    )


def scenario_depot_location_and_utilization(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> List[folium.Marker]:
    """
    Get information about the depots in a given scenario.

    :param scenario:
    :param session:
    :return:
    """
    depots = session.query(Depot).filter(Depot.scenario_id == scenario.id).all()
    result = []
    for depot in tqdm(depots, desc="Depots"):
        result.append(
            single_depot_location_and_utilization(scenario, session, depot.id)
        )

    return result


def mean_location(session: sqlalchemy.orm.session.Session) -> Dict[str, float]:
    """
    Get the mean location of all charging stations.

    :param session:
    :param station_ids:
    :return:
    """
    stations = session.query(Station).filter(Station.is_electrified == True).all()
    latitudes = []
    longitudes = []
    for station in stations:
        point = geoalchemy2.shape.to_shape(station.geom)
        latitudes.append(point.y)
        longitudes.append(point.x)

    return {"latitude": np.mean(latitudes), "longitude": np.mean(longitudes)}


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    SCENARIO_NAMES = ["OU", "DEP", "TERM"]

    # Create a folium map
    mean_location_dict = mean_location(session)
    m = folium.Map(
        location=[mean_location_dict["latitude"], mean_location_dict["longitude"]],
        zoom_start=11,
    )

    feature_groups: Dict[str, List[folium.FeatureGroup]] = dict()

    for scenario_name in SCENARIO_NAMES:
        # Create a FeatureGroup. This will enable the user to toggle the visibility of the markers
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )
        feature_group_term = folium.FeatureGroup(name=scenario.name)
        # Get the information about the charging stations
        stations = scenario_charging_station_location_and_utilization(scenario, session)
        for station in stations:
            feature_group_term.add_child(station)
        if "Endhaltestellen aus Szenario" not in feature_groups.keys():
            feature_groups["Endhaltestellen aus Szenario"] = []
        feature_groups["Endhaltestellen aus Szenario"].append(feature_group_term)
        m.add_child(feature_group_term)

        feature_group_depot = folium.FeatureGroup(name=scenario.name)
        # Get the information about the depots
        depots = scenario_depot_location_and_utilization(scenario, session)
        for depot in depots:
            feature_group_depot.add_child(depot)
        if "Depots aus Szenario" not in feature_groups.keys():
            feature_groups["Depots aus Szenario"] = []
        feature_groups["Depots aus Szenario"].append(feature_group_depot)
        m.add_child(feature_group_depot)

    # Add a grouped LayerControl for the termini
    grouped_layer_control_termini = folium.plugins.GroupedLayerControl(
        feature_groups,
        collapsed=False,
    )
    m.add_child(grouped_layer_control_termini)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    m.save(os.path.join(OUTPUT_FOLDER, "index.html"))
