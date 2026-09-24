"""
case03_config.py
Confined Fairway Crossing Give-Way with Turning TS.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# Fairway along X between Y in [-5.0, 5.0], with an opening on East side for crossing
poly_west = Polygon([
    (-40.0, -25.0),
    (40.0, -25.0),
    (40.0, -4.0),
    (-40.0, -4.0)
])

poly_east_south = Polygon([
    (-40.0, 4.0),
    (-8.0, 4.0),
    (-8.0, 35.0),
    (-40.0, 35.0)
])

poly_east_north = Polygon([
    (8.0, 4.0),
    (40.0, 4.0),
    (40.0, 35.0),
    (8.0, 35.0)
])

poly_canal_full = so.unary_union([poly_west, poly_east_south, poly_east_north])

CASE03_CONFIG = {
    "name": "case03",
    "description": "Akdag Confined Fairway Crossing Give-Way with Turning TS (Out of Range Start)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -4.0,
        "y_max": 4.0,
        "x_min": -35.0,
        "x_max": 35.0
    },

    # Own Ship starts south at x = -30.0 m
    "os_initial_state": np.array([-30.0, 0.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.45,
    "os_mission_wps": np.array([
        [-30.0, 0.0],
        [ 30.0, 0.0]
    ], dtype=np.float64),

    # Target Ship starts East at y = +32.0 m
    # Initial separation = 43.8 m (Crossing R_IS = 34.9 m -> Starts safely out of range)
    "ts_initial_state": np.array([0.0, 32.0, -np.pi / 2.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,
    "ts_mission_wps": np.array([
        [ 0.0, 32.0],
        [ 0.0,  2.5],
        [30.0,  2.5]
    ], dtype=np.float64),
    "t_sim": 180.0
}