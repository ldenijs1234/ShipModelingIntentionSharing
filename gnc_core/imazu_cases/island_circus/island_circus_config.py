"""
island_circus_config.py
Configuration parameters for Island Circus scenario (Slide 1 Special).
Central island at (0,0) with OS and TS sailing confrontational arcs around the fairway.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon

# 1. Central Island Shoreline Geometry (Radius = 12.0 m)
angles = np.linspace(0, 2.0 * np.pi, 36, endpoint=False)
r_inner = 12.0
island_x = r_inner * np.cos(angles)
island_y = r_inner * np.sin(angles)
poly_island = Polygon(np.column_stack([island_x, island_y]))

# 2. Channel Parameters
r_mid = 20.0     # Mid-channel sailing line radius [m]
u_nominal = 1.2  # Nominal surge velocity [m/s]

# 3. OS Waypoints: Clockwise arc on East side from North (20, 0) -> East (0, 20) -> South (-20, 0)
angles_os = np.linspace(0.0, np.pi, 15)
os_x = r_mid * np.cos(angles_os)
os_y = r_mid * np.sin(angles_os)
os_wps = np.column_stack([os_x, os_y])

# 4. TS Waypoints: Counter-Clockwise arc on East side from South (-20, 0) -> East (0, 22) -> North (20, 0)
angles_ts = np.linspace(np.pi, 0.0, 15)
ts_x = r_mid * np.cos(angles_ts)
ts_y = r_mid * np.sin(angles_ts)
ts_wps = np.column_stack([ts_x, ts_y])

# 5. Initial headings (psi) in NED
psi_os_init = float(np.arctan2(os_wps[1, 1] - os_wps[0, 1], os_wps[1, 0] - os_wps[0, 0]))
psi_ts_init = float(np.arctan2(ts_wps[1, 1] - ts_wps[0, 1], ts_wps[1, 0] - ts_wps[0, 0]))

ISLAND_CIRCUS_CONFIG = {
    "name": "island_circus",
    "description": "Island Circus: Dynamic Head-On Encounter around a Central Island",

    "canal_polygons": poly_island,
    "canal_bounds": {
        "y_min": -32.0,
        "y_max": 32.0,
        "x_min": -32.0,
        "x_max": 32.0
    },
    "canal_width": 16.0,

    # Own Ship (OS) Initial Conditions & Route
    "os_initial_state": np.array([os_wps[0, 0], os_wps[0, 1], psi_os_init, u_nominal, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": u_nominal,
    "os_mission_wps": os_wps,

    # Target Ship (TS) Initial Conditions & Route
    "ts_initial_state": np.array([ts_wps[0, 0], ts_wps[0, 1], psi_ts_init, u_nominal, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": u_nominal,
    "ts_mission_wps": ts_wps,
    "t_sim": 60.0
}