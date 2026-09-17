from typing import Optional
import numpy as np


class StateEstimation:

    _last_known_ais: Optional[np.ndarray] = None
    _extrapolated_state: Optional[np.ndarray] = None

    @classmethod
    def reset(cls) -> None:
        """Clears cached extrapolation states between simulation runs."""
        cls._last_known_ais = None
        cls._extrapolated_state = None

    @staticmethod
    def estimate_own_state(raw_state: np.ndarray) -> np.ndarray:
        """Maps internal state [X, Y, psi, r, b, u] to [X, Y, psi, u, v, r] with v=0 (Eq. 3.8)."""
        X, Y, psi, r, _, u = raw_state
        return np.array([X, Y, psi, u, 0.0, r], dtype=np.float64)

    @classmethod
    def estimate_target_from_ais(
        cls,
        ais_current: Optional[np.ndarray],
        ais_prev: Optional[np.ndarray] = None,
        dt_ais: float = 0.05,
    ) -> np.ndarray:
        """
        Reconstructs the 6-element TS state vector [X, Y, psi, u, v, r].
        Dead-reckons forward when AIS packets arrive at discrete intervals (e.g. 3.3s - 10s).
        """
        if ais_current is None or len(ais_current) < 4:
            if cls._extrapolated_state is not None:
                psi = cls._extrapolated_state[2]
                r = cls._extrapolated_state[5]
                u = cls._extrapolated_state[3]

                # Advance kinematics along the curve
                cls._extrapolated_state[2] = (psi + r * dt_ais + np.pi) % (2.0 * np.pi) - np.pi
                cls._extrapolated_state[0] += u * np.cos(cls._extrapolated_state[2]) * dt_ais
                cls._extrapolated_state[1] += u * np.sin(cls._extrapolated_state[2]) * dt_ais
                return np.copy(cls._extrapolated_state)
            return np.zeros(6, dtype=np.float64)

        # Detect discrete AIS arrival
        is_fresh_packet = False
        if cls._last_known_ais is None:
            is_fresh_packet = True
        else:
            dx = ais_current[0] - cls._last_known_ais[0]
            dy = ais_current[1] - cls._last_known_ais[1]
            if np.hypot(dx, dy) > 1e-4:
                is_fresh_packet = True

        if is_fresh_packet or cls._extrapolated_state is None:
            cls._last_known_ais = np.copy(ais_current)
            if len(ais_current) >= 6:
                X, Y, psi, u_ts = ais_current[0], ais_current[1], ais_current[2], ais_current[3]
                r_ts = ais_current[5]
            else:
                X, Y, psi, u_ts = ais_current[:4]
                r_ts = 0.0

            cls._extrapolated_state = np.array(
                [X, Y, psi, float(u_ts), 0.0, float(r_ts)], dtype=np.float64
            )
        else:
            # Dead reckoning between discrete AIS updates
            psi = cls._extrapolated_state[2]
            r = cls._extrapolated_state[5]
            u = cls._extrapolated_state[3]

            cls._extrapolated_state[2] = (psi + r * dt_ais + np.pi) % (2.0 * np.pi) - np.pi
            cls._extrapolated_state[0] += u * np.cos(cls._extrapolated_state[2]) * dt_ais
            cls._extrapolated_state[1] += u * np.sin(cls._extrapolated_state[2]) * dt_ais

        return np.copy(cls._extrapolated_state)