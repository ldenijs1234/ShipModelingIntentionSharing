"""
case02_config.py
Confined Fairway Encounter with Turning TS.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-45.0, -18.0),
    (45.0, -18.0),
    (45.0, -4.0),
    (-45.0, -4.0)
])

poly_starboard = Polygon([
    (-45.0, 15.0),
    (45.0, 15.0),
    (45.0, 4.0),
    (-45.0, 4.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE02_CONFIG = {
    "name": "case02",
    "description": "Akdag Confined Fairway Encounter with Turning TS (Out of Range Start)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -40.0,
        "x_max": 40.0
    },

    # Own Ship (Blue) starts further south at x = -30.0 m
    "os_initial_state": np.array([-30.0, 0.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.45,
    "os_mission_wps": np.array([
        [-30.0, 0.0],
        [-10.0, 0.0],
        [ -2.0, 0.0],
        [  5.0, 0.0],
        [ 15.0, 0.0],
        [ 30.0, 0.0]
    ], dtype=np.float64),

    # Target Ship starts further north at x = +30.0 m
    # Initial separation = 60.0 m (R_IS = 49.3 m -> Starts safely out of range)
    "ts_initial_state": np.array([30.0, -1.0, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [ 30.0, -1.0],
        [  6.0, -1.0],
        [-30.0, 10.0]
    ], dtype=np.float64),
    "t_sim": 180.0
}