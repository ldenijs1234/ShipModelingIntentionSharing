from typing import Tuple, Optional
import numpy as np
from gnc_core.config.vessel_params import VesselParams
from gnc_core.navigation.risk import RiskCalculator


class DecisionLayer:

    @staticmethod
    def evaluate(
        x_os: np.ndarray,
        x_ts: np.ndarray,
        w_os: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        dcpa: float,
        tcpa: float,
    ) -> Tuple[np.ndarray, float, str]:
        """Evaluates logic tree (Figure 3.5) to output active waypoints or reactive heading offset."""
        risk = (dcpa <= VesselParams.DCPA_safe) and (0.0 <= tcpa <= VesselParams.TCPA_safe)

        # Mode A: Shared intention available
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            if risk:
                w_evasive = np.copy(w_os)
                for i in range(min(3, len(w_evasive))):
                    w_evasive[i, 1] += 1.2 * VesselParams.DCPA_safe
                return w_evasive, 0.0, "State A.1"
            return np.copy(w_os), 0.0, "State A.2"

        # Mode B: Uncollaborative straight-line fallback
        if risk:
            beta = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta)
            offset = 45.0 if scenario in ["Head-On", "Crossing_A"] else (30.0 if scenario == "Overtaking" else 0.0)
            return np.copy(w_os), float(np.radians(offset)), "State B.1"

        return np.copy(w_os), 0.0, "State B.2"