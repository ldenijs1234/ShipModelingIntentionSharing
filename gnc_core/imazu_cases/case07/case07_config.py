"""
case07_config.py
Inland Canal Dynamic Intent Modification Encounter (Stale Intent / Bait-and-Switch).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

poly_port = Polygon([
    (-45.0, -15.0),
    (75.0, -15.0),
    (75.0, -3.0),
    (-45.0, -3.0)
])

poly_starboard = Polygon([
    (-45.0, 3.0),
    (75.0, 3.0),
    (75.0, 15.0),
    (-45.0, 15.0)
])

poly_canal_full = so.unary_union([poly_port, poly_starboard])

CASE07_CONFIG = {
    "name": "case07",
    "description": "Dynamic Intent Modification (Stale Intent Under Latency & Interval Degradation)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -6.0,
        "y_max": 6.0,
        "x_min": -40.0,
        "x_max": 70.0
    },

    # Own Ship (OS starts at X = -35.0 m, heading North along Y = 1.0 m)
    "os_initial_state": np.array([-35.0, 1.0, 0.0, 0.52, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.52,
    "os_mission_wps": np.array([
        [-35.0, 1.0],
        [ 65.0, 1.0]
    ], dtype=np.float64),

    # Target Ship Initial State (Approaching Head-On / Angled from X = 35.0 m)
    "ts_initial_state": np.array([35.0, -1.5, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.45,

    # 1. Initial nominal intent transmitted at start (appears to clear OS safely on port)
    "ts_mission_wps": np.array([
        [ 35.0, -1.5],
        [ 10.0, -1.5],
        [-35.0, -1.5]
    ], dtype=np.float64),

    # 2. Dynamic Update Event: Mid-run revision
    "dynamic_event": {
        "trigger_type": "time",     # Can trigger on simulation time or TS x-position
        "trigger_value": 45.0,      # Trigger at t = 45.0 s (or e.g. x_ts <= 12.0 m)
        "revised_ts_wps": np.array([
            [ 10.0, -1.5],
            [ -2.0,  0.8],          # Cuts sharply into OS's lane
            [-35.0,  1.0]
        ], dtype=np.float64)
    },

    "t_sim": 180.0
}