import numpy as np


class StateEstimation:

    @staticmethod
    def estimate_own_state(raw_state: np.ndarray) -> np.ndarray:
        """Maps internal state [X, Y, psi, r, b, u] to [X, Y, psi, u, v, r] with v=0 (Eq. 3.8)."""
        X, Y, psi, r, _, u = raw_state
        return np.array([X, Y, psi, u, 0.0, r], dtype=np.float64)

    @staticmethod
    def estimate_target_from_ais(
        ais_current: np.ndarray,
        ais_prev: np.ndarray | None = None,
        dt_ais: float = 0.02,
    ) -> np.ndarray:
        """
        Reconstructs 6-state vector X_TS from raw AIS inputs [X, Y, psi, U].
        Handles either a 4-element AIS input [X, Y, psi, U] or a 6-element input.
        """
        if len(ais_current) >= 6:
            X, Y, psi, u_ts = ais_current[0], ais_current[1], ais_current[2], ais_current[3]
            r_ts = ais_current[5]
            return np.array([X, Y, psi, u_ts, 0.0, r_ts], dtype=np.float64)

        X, Y, psi, U = ais_current[:4]
        if ais_prev is not None and dt_ais > 1e-4:
            psi_prev = ais_prev[2]
            d_psi = (psi - psi_prev + np.pi) % (2.0 * np.pi) - np.pi
            r_est = float(d_psi / dt_ais)
        else:
            r_est = 0.0

        return np.array([X, Y, psi, float(U), 0.0, r_est], dtype=np.float64)