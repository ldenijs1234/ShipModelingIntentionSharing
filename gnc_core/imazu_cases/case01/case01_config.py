"""
case01_config.py
Configuration parameters for Imazu Case 01 (Head-On Encounter).
Coordinates are in NED [North (X), East (Y)].
"""

import numpy as np

CASE01_CONFIG = {
    "name": "case01",
    "description": "Imazu Case 01: Symmetrical Head-On Encounter",
    
    # Own Ship (OS) Initial Conditions & Route
    # State: [X (North), Y (East), psi (rad), r, b, u]
    "os_initial_state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64),
    "os_nominal_speed": 0.5,
    "os_mission_wps": np.array([
        [0.0, 0.0],
        [40.0, 0.0]
    ], dtype=np.float64),

    # Target Ship (TS) Initial Conditions & Route
    # Starting at x=25.0, heading south (psi = pi) toward origin
    "ts_initial_state": np.array([25.0, 0.0, np.pi, 0.0, 0.0, 0.5], dtype=np.float64),
    "ts_nominal_speed": 0.5,
    "ts_mission_wps": np.array([
        [25.0, 0.0],
        [12.5, 0.0],
        [0.0, 0.0]
    ], dtype=np.float64),
}