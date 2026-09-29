"""
case07_config.py
Inland Canal Curvilinear Bend (Opposing Head-on Transit, Flipped over X-axis into -Y).
Both vessels keep to their lawful Starboard side (COLREGs Rule 9):
- OS starts South at X = -35.0 m, heading North along Starboard lane (+Y).
- TS starts North at the far end of the bend, heading South along its Starboard lane (which is -Y relative to northbound centerline).
Waypoint routes are swapped and oriented for opposing transit, with a safe curvilinear clearance of 1.30 m.
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so

# -------------------------------------------------------------------------
# 1. Fairway Centerline Geometry (Curving into -Y / Port)
# -------------------------------------------------------------------------
L_SEG = 14.0
angles = np.radians([0.0, -15.0, -30.0, -45.0, -60.0])
centerline_pts = [[-45.0, 0.0], [-10.0, 0.0]]
x_curr, y_curr = -10.0, 0.0

for ang in angles[1:]:
    x_curr += L_SEG * np.cos(ang)
    y_curr += L_SEG * np.sin(ang)
    centerline_pts.append([x_curr, y_curr])

centerline_pts.append([x_curr + 30.0 * np.cos(angles[-1]), y_curr + 30.0 * np.sin(angles[-1])])
centerline = np.array(centerline_pts, dtype=np.float64)

# -------------------------------------------------------------------------
# 2. Compute Normal Vectors along Centerline
# -------------------------------------------------------------------------
N = len(centerline)
normals = np.zeros((N, 2), dtype=np.float64)

for i in range(N):
    if i == 0:
        dx = centerline[1, 0] - centerline[0, 0]
        dy = centerline[1, 1] - centerline[0, 1]
    elif i == N - 1:
        dx = centerline[-1, 0] - centerline[-2, 0]
        dy = centerline[-1, 1] - centerline[-2, 1]
    else:
        dx = centerline[i + 1, 0] - centerline[i - 1, 0]
        dy = centerline[i + 1, 1] - centerline[i - 1, 1]

    norm = np.hypot(dx, dy)
    normals[i] = [dy / norm, -dx / norm]  # Starboard (+Y) normal vector

# -------------------------------------------------------------------------
# 3. Explicit Canal Bank Obstacle Polygons
# -------------------------------------------------------------------------
W_FAIRWAY = 3.0  # Fairway half-width: 6.0 m total channel width
W_BANK = 15.0

port_inner = centerline - W_FAIRWAY * normals
port_outer = centerline - (W_FAIRWAY + W_BANK) * normals

stbd_inner = centerline + W_FAIRWAY * normals
stbd_outer = centerline + (W_FAIRWAY + W_BANK) * normals

poly_port = Polygon(np.vstack([port_inner, port_outer[::-1]]))
poly_starboard = Polygon(np.vstack([stbd_inner, stbd_outer[::-1]]))
poly_canal_full = so.unary_union([poly_port, poly_starboard])

# -------------------------------------------------------------------------
# 4. Swapped Waypoint Routes & Opposing Initial Positions
# -------------------------------------------------------------------------
# OS sails North on Starboard side (+0.65 m normal offset)
os_mission_wps = centerline - 0.65 * normals

# TS sails South on its Starboard side (which corresponds to -0.65 m normal offset)
# Reverse coordinates so TS navigates from far bend back to South
ts_mission_wps = (centerline + 0.65 * normals)[::-1].copy()

# Initial Heading for TS pointing back along its first path segment
ts_initial_pos = ts_mission_wps[0]
ts_initial_heading = np.arctan2(
    ts_mission_wps[1, 1] - ts_mission_wps[0, 1],
    ts_mission_wps[1, 0] - ts_mission_wps[0, 0]
)

CASE07_CONFIG = {
    "name": "case07",
    "description": "Opposing Bend Encounter: Swapped Routes, Both on Lawful Starboard Side (Rule 9)",

    "canal_polygons": poly_canal_full,
    "canal_bounds": {
        "y_min": -45.0,
        "y_max": 15.0,
        "x_min": -50.0,
        "x_max": 70.0
    },

    # Own Ship (OS): Starts South at X = -35.0 m, sails North along Starboard lane (+Y)
    "os_initial_state": np.array([-35.0, os_mission_wps[0, 1], 0.0, 0.52, 0.0, 0.0], dtype=np.float64),
    "os_nominal_speed": 0.52,
    "os_mission_wps": os_mission_wps,

    # Target Ship (TS): Starts North beyond bend, sails South along its Starboard lane (-Y)
    # Lateral clearance between curved paths is 1.30 m (> d_safe = 1.0 m) throughout
    "ts_initial_state": np.array([ts_initial_pos[0], ts_initial_pos[1], ts_initial_heading, 0.52, 0.0, 0.0], dtype=np.float64),
    "ts_nominal_speed": 0.52,
    "ts_mission_wps": ts_mission_wps,

    "t_sim": 140.0
}