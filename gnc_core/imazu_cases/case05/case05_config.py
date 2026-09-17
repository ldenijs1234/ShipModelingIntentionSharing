import numpy as np
from shapely.geometry import Polygon, MultiPolygon

# Canal boundaries (River Nieuwe Maas + Rijnhaven harbor basin)
west_bank = Polygon([
    [-30.0, -10.0], 
    [-10.0, -10.0], 
    [10.0, -12.0], 
    [35.0, -18.0], 
    [35.0, -40.0], 
    [-30.0, -40.0]
])

east_bank_north = Polygon([
    [8.0, 10.0], 
    [20.0, 12.0], 
    [35.0, 16.0], 
    [35.0, 40.0], 
    [8.0, 40.0]
])

east_bank_south = Polygon([
    [-30.0, 10.0], 
    [-8.0, 10.0], 
    [-8.0, 40.0], 
    [-30.0, 40.0]
])

canal_multipolygon = MultiPolygon([west_bank, east_bank_north, east_bank_south])

CASE05_CONFIG = {
    # ---------------------------------------------------------------------
    # Own Ship (OS) - Sailing North along the curved river
    # Format: [X, Y, psi, u, v, r]
    # ---------------------------------------------------------------------
    "os_initial_state": np.array([-12.0, 2.0, 0.0, 0.45, 0.0, 0.0], dtype=np.float64),
    "os_mission_wps": np.array([
        [-12.0, 2.0],  # Start
        [0.0, 2.0],    # Approach intersection
        [15.0, 4.0],   # Follow river curve
        [30.0, 8.0]    # End mission
    ], dtype=np.float64),
    "os_nominal_speed": 0.45,

    # ---------------------------------------------------------------------
    # Target Ship (TS) - Sailing South, turning East into the harbor
    # ---------------------------------------------------------------------
    "ts_initial_state": np.array([16.0, -2.0, np.pi, 0.45, 0.0, 0.0], dtype=np.float64),
    "ts_mission_wps": np.array([
        [16.0, -2.0],  # Start
        [5.0, -2.0],   # Turning point
        [-2.0, 15.0],  # Entering harbor basin
        [-2.0, 25.0]   # Docked
    ], dtype=np.float64),
    "ts_nominal_speed": 0.45,

    # ---------------------------------------------------------------------
    # Environment & Simulation Parameters
    # ---------------------------------------------------------------------
    "canal_polygons": canal_multipolygon,
    "t_sim": 180.0
}