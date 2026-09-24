"""
case05_config.py
Harbor turn crossing scenario (Trimmed Exit).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

west_bank = Polygon([
    [-35.0, -10.0], 
    [-10.0, -10.0], 
    [ 10.0, -12.0], 
    [ 25.0, -18.0], 
    [ 25.0, -40.0], 
    [-35.0, -40.0]
])

east_bank_north = Polygon([
    [ 8.0, 10.0], 
    [20.0, 12.0], 
    [25.0, 16.0], 
    [25.0, 25.0], 
    [ 8.0, 25.0]
])

east_bank_south = Polygon([
    [-35.0, 10.0], 
    [ -8.0, 10.0], 
    [ -8.0, 25.0], 
    [-35.0, 25.0]
])

canal_multipolygon = MultiPolygon([west_bank, east_bank_north, east_bank_south])

CASE05_CONFIG = {
    "name": "case05",
    "description": "Harbor turn encounter with curved channel (Trimmed Exit)",

    # OS starts at x = -26.0 m, finishes at x = 14.0 m
    "os_initial_state": np.array([-26.0, 2.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_mission_wps": np.array([
        [-26.0, 2.0],
        [  0.0, 2.0],
        [ 14.0, 4.0]
    ], dtype=np.float64),
    "os_nominal_speed": 0.45,

    # TS starts at x = 30.0 m, docks in harbor at y = 10.0 m
    "ts_initial_state": np.array([30.0, -2.0, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_mission_wps": np.array([
        [ 30.0, -2.0],
        [  5.0, -2.0],
        [ -2.0, 10.0]
    ], dtype=np.float64),
    "ts_nominal_speed": 0.45,

    "canal_polygons": canal_multipolygon,
    "t_sim": 95.0
}