from typing import Tuple, Dict, Any, Optional
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
        tick: int,
        cached: Dict[str, Any],
        dt: float = 0.02,
        u_nominal: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Executes one tick (dt = 0.02s) of the multi-rate synchronous loop (Section 3.2.3).
        - 50 Hz: State estimation, control, plant integration (k % 1 == 0)
        - 20 Hz: Guidance (k % 2 == 0 or sub-stepped)
        - 10 Hz: Risk & Decision (k % 5 == 0)
        """
        # 1. State Estimation (50 Hz)
        x_os = StateEstimation.estimate_own_state(internal_state)
        x_ts = StateEstimation.estimate_target_state(x_ts_raw)

        # 2. Risk & Decision (10 Hz: every 5 ticks)
        if tick % 5 == 0:
            cached["dcpa"], cached["tcpa"] = RiskCalculator.calculate_cpa(x_os, x_ts)
            cached["w_active"], cached["psi_ca"], cached["state"] = DecisionLayer.evaluate(
                x_os, x_ts, w_mission_os, w_ts_delayed, cached["dcpa"], cached["tcpa"]
            )

        # 3. Guidance Layer (20 Hz: every 2 ticks)
        if tick % 2 == 0:
            cached["psi_wp"], _, cached["wp_idx"] = LOSGuidance.compute_heading_reference(
                x_os, cached["w_active"], cached["wp_idx"]
            )

        # 4. Control Layer (50 Hz)
        u_c, tau_c, psi_cmd = Autopilot.compute_control(
            x_os, cached["psi_wp"], cached["psi_ca"], u_nominal
        )

        # 5. Vessel Dynamics Integration via RK4 (50 Hz)
        next_internal_state = VesselDynamics.rk4(internal_state, np.array([tau_c, u_c]), dt=dt)

        telemetry = {
            "time": tick * dt,
            "x_os": x_os,
            "u_c": u_c,
            "tau_c": tau_c,
            "psi_cmd": psi_cmd,
            "dcpa": cached["dcpa"],
            "tcpa": cached["tcpa"],
            "active_state": cached["state"],
        }
        return next_internal_state, cached, telemetry