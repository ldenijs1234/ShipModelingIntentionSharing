"""
case03_config.py
Confined Fairway Crossing Give-Way with Turning TS (Synchronized Intersection).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_west = Polygon([
    (-30.0, -25.0),
    (20.0, -25.0),
    (20.0, -4.0),
    (-30.0, -4.0)
])

poly_east_south = Polygon([
    (-30.0, 4.0),
    (-8.0, 4.0),
    (-8.0, 30.0),
    (-30.0, 30.0)
])

poly_east_north = Polygon([
    (8.0, 4.0),
    (20.0, 4.0),
    (20.0, 30.0),
    (8.0, 30.0)
])

poly_canal_full = so.unary_union([poly_west, poly_east_south, poly_east_north])

CASE03_CONFIG = {
    "name": "case03",
    "description": "Akdag Confined Fairway Crossing Give-Way with Turning TS (Synchronized Intersection)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -30.0,
        "x_max": 20.0
    },

    # Own Ship (Blue) starts South at X = -25.0 m, finishes at X = 12.0 m
    # Reaches (0,0) in exactly 55.5 s
    "os_initial_state": np.array([-25.0, 0.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.45,
    "os_mission_wps": np.array([
        [-25.0, 0.0],
        [ 12.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (Red) starts East at Y = 25.0 m, crosses to (0,0) before turning
    # Reaches (0,0) in exactly 55.5 s -> Direct simultaneous collision path
    "ts_initial_state": np.array([0.0, 25.0, -np.pi / 2.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [ 0.0, 25.0],
        [ 0.0,  0.0],   # Crosses right through the centerline conflict point
        [12.0,  2.5]    # Turns North along the East fairway lane
    ], dtype=np.float64),
    "t_sim": 85.0
}