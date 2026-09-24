"""
case01_config.py
Configuration parameters for Imazu Case 01 (Head-On Encounter) with Canal Boundaries.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# Define canal banks (fairway width = 6.0m, Y between -3.0 and +3.0)
# Extended in X from -40.0 to +75.0 to accommodate the expanded head-on start
poly_port = Polygon([
    (-40.0, -15.0),
    (75.0, -15.0),
    (75.0, -3.0),
    (-40.0, -3.0)
])

poly_starboard = Polygon([
    (-40.0, 3.0),
    (75.0, 3.0),
    (75.0, 15.0),
    (-40.0, 15.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE01_CONFIG = {
    "name": "case01",
    "description": "Imazu Case 01: Symmetrical Head-On Encounter with Canal Walls (Out of Range Start)",
    
    # Canal polygon definition
    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -3.0,
        "y_max": 3.0,
        "x_min": -35.0,
        "x_max": 70.0
    },

    # Own Ship (OS) Initial Conditions & Route
    # Starts at x = -20.0 m, heading North
    "os_initial_state": np.array([-20.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.5,
    "os_mission_wps": np.array([
        [-20.0, 0.0],
        [ 50.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (TS) Initial Conditions & Route
    # Starts at x = +45.0 m, heading South (pi rad)
    # Initial separation = 65.0 m (R_IS = 54.8 m -> Starts safely out of range)
    "ts_initial_state": np.array([45.0, 0.0, np.pi, 0.5, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.5,
    "ts_mission_wps": np.array([
        [ 45.0, 0.0],
        [ 12.5, 0.0],
        [-25.0, 0.0]
    ], dtype=np.float64),
    "t_sim": 180.0
}