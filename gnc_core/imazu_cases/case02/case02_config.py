"""
case02_config.py
Confined Fairway Encounter with Turning TS (Trimmed Exit).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-35.0, -18.0),
    (35.0, -18.0),
    (35.0, -4.0),
    (-35.0, -4.0)
])

poly_starboard = Polygon([
    (-35.0, 15.0),
    (35.0, 15.0),
    (35.0, 4.0),
    (-35.0, 4.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE02_CONFIG = {
    "name": "case02",
    "description": "Akdag Confined Fairway Encounter with Turning TS (Trimmed Exit)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -35.0,
        "x_max": 35.0
    },

    # Own Ship (Blue) finishes at X = 14.0 m
    "os_initial_state": np.array([-30.0, 0.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.45,
    "os_mission_wps": np.array([
        [-30.0, 0.0],
        [-10.0, 0.0],
        [ -2.0, 0.0],
        [  5.0, 0.0],
        [ 14.0, 0.0]
    ], dtype=np.float64),

    # Target Ship ends shortly after clearing the channel at X = -8.0 m
    "ts_initial_state": np.array([30.0, -1.0, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [ 30.0, -1.0],
        [  6.0, -1.0],
        [ -8.0,  5.0]
    ], dtype=np.float64),
    "t_sim": 95.0
}