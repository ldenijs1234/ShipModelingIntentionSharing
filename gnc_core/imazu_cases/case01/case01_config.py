"""
case01_config.py
Configuration parameters for Imazu Case 01 (Head-On Encounter) with Canal Boundaries.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-30.0, -15.0),
    (55.0, -15.0),
    (55.0, -3.0),
    (-30.0, -3.0)
])

poly_starboard = Polygon([
    (-30.0, 3.0),
    (55.0, 3.0),
    (55.0, 15.0),
    (-30.0, 15.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE01_CONFIG = {
    "name": "case01",
    "description": "Imazu Case 01: Symmetrical Head-On Encounter with Canal Walls (Trimmed Exit)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -3.0,
        "y_max": 3.0,
        "x_min": -25.0,
        "x_max": 50.0
    },

    # Own Ship (OS) Initial Conditions & Route
    "os_initial_state": np.array([-20.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.5,
    "os_mission_wps": np.array([
        [-20.0, 0.0],
        [ 25.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (TS) Initial Conditions & Route
    "ts_initial_state": np.array([45.0, 0.0, np.pi, 0.5, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.5,
    "ts_mission_wps": np.array([
        [45.0, 0.0],
        [12.5, 0.0],
        [ 0.0, 0.0]
    ], dtype=np.float64),
    "t_sim": 85.0
}