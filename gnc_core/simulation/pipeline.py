from typing import Any, Dict, Optional, Tuple
import numpy as np

from gnc_core.models.vessel_dynamics import VesselDynamics
from gnc_core.navigation.state_estimation import StateEstimation
from gnc_core.navigation.risk import RiskCalculator
from gnc_core.guidance.decision import DecisionLayer
from gnc_core.guidance.los import LOSGuidance
from gnc_core.control.autopilot import Autopilot


class SynchronousPipeline:

    @staticmethod
    def step(
        internal_state: np.ndarray,
        w_mission_os: np.ndarray,
        x_ts_raw: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        cached: Dict[str, Any],
        dt: float = 0.1,       # Default strictly to 10 Hz
        u_nominal: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        
        # 1. State Estimation (10 Hz)
        x_os = StateEstimation.estimate_own_state(internal_state)
        x_ts = StateEstimation.estimate_target_from_ais(x_ts_raw, dt_ais=dt)

        # 2. Risk & Decision (10 Hz)
        cached["dcpa"], cached["tcpa"] = RiskCalculator.calculate_cpa(x_os, x_ts)
        cached["w_active"], cached["psi_ca"], cached["state"] = DecisionLayer.evaluate(
            x_os, x_ts, w_mission_os, w_ts_delayed, cached["dcpa"], cached["tcpa"]
        )

        # 3. Guidance Layer (10 Hz)
        cached["psi_wp"], _, cached["wp_idx"] = LOSGuidance.compute_heading_reference(
            x_os, cached["w_active"], cached["wp_idx"]
        )

        # 4. Control Layer (10 Hz)
        u_c, tau_c, psi_cmd = Autopilot.compute_control(
            x_os, cached["psi_wp"], cached["psi_ca"], u_nominal
        )

        # 5. Vessel Dynamics Integration via RK4 (10 Hz)
        next_internal_state = VesselDynamics.rk4(internal_state, np.array([tau_c, u_c]), dt=dt)

        telemetry = {
            "time": cached.get("time", 0.0) + dt,
            "x_os": x_os,
            "u_c": u_c,
            "tau_c": tau_c,
            "psi_cmd": psi_cmd,
            "dcpa": cached["dcpa"],
            "tcpa": cached["tcpa"],
            "active_state": cached["state"],
        }
        cached["time"] = telemetry["time"]
        
        return next_internal_state, cached, telemetry