"""
case05_config.py
Harbor turn crossing scenario.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

west_bank = Polygon([
    [-45.0, -10.0], 
    [-10.0, -10.0], 
    [ 10.0, -12.0], 
    [ 40.0, -18.0], 
    [ 40.0, -40.0], 
    [-45.0, -40.0]
])

east_bank_north = Polygon([
    [ 8.0, 10.0], 
    [20.0, 12.0], 
    [40.0, 16.0], 
    [40.0, 40.0], 
    [ 8.0, 40.0]
])

east_bank_south = Polygon([
    [-45.0, 10.0], 
    [ -8.0, 10.0], 
    [ -8.0, 40.0], 
    [-45.0, 40.0]
])

canal_multipolygon = MultiPolygon([west_bank, east_bank_north, east_bank_south])

CASE05_CONFIG = {
    "name": "case05",
    "description": "Harbor turn encounter with curved channel (Out of Range Start)",

    # OS starts further south at x = -26.0 m
    # TS starts further north at x = +28.0 m
    # Initial separation = 54.1 m (Head-on closing R_IS = 49.3 m -> Starts safely out of range)
    "os_initial_state": np.array([-26.0, 2.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_mission_wps": np.array([
        [-26.0, 2.0],   # Start
        [  0.0, 2.0],   # Approach intersection
        [ 15.0, 4.0],   # Follow river curve
        [ 35.0, 8.0]    # End mission
    ], dtype=np.float64),
    "os_nominal_speed": 0.45,

    "ts_initial_state": np.array([30.0, -2.0, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_mission_wps": np.array([
        [ 30.0, -2.0],  # Start
        [  5.0, -2.0],  # Turning point
        [ -2.0, 15.0],  # Entering harbor basin
        [ -2.0, 25.0]   # Docked
    ], dtype=np.float64),
    "ts_nominal_speed": 0.45,

    "canal_polygons": canal_multipolygon,
    "t_sim": 180.0
}