"""
SPECTER TDOA Geolocation Simulator.

⚠  ALL FUNCTIONS IN THIS MODULE ARE SIMULATED.
   No real RF hardware, real receivers, or real propagation physics are used.
   Time delays are derived analytically from Euclidean geometry and then
   corrupted with synthetic Gaussian noise to approximate real-world error.

TDOA (Time Difference of Arrival) principle:
    A transmitter emits a signal that arrives at each receiver at a slightly
    different time proportional to the distance travelled.  The difference in
    arrival times (TDOA) between pairs of receivers constrains the transmitter
    to lie on a hyperbola.  With ≥ 3 receivers we get ≥ 2 independent TDOAs
    whose hyperbola intersection pins down the 2-D position.

Solver:
    We use a least-squares formulation (scipy.optimize.least_squares) rather
    than closed-form hyperbolic algebra.  This generalises gracefully to more
    than 3 receivers and handles noisy inputs without singularities.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; override before import pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import least_squares
from typing import Optional


# Speed of light / radio propagation speed (m/s)
_C = 3.0e8

# Default receiver triangle: equilateral, 1 km side, centred at origin (metres)
_DEFAULT_RECEIVERS = [
    (   0.0,  577.35),   # top
    (-500.0, -288.68),   # bottom-left
    ( 500.0, -288.68),   # bottom-right
]


# ---------------------------------------------------------------------------
# TDOA Simulator
# ---------------------------------------------------------------------------

class TDOASimulator:
    """
    ⚠ SIMULATED — Synthetic TDOA-based transmitter geolocation.

    Simulates a 3-receiver passive listening array that measures the
    difference in signal arrival times to triangulate a transmitter position.

    All coordinates are in metres relative to an arbitrary local origin.

    Args:
        receiver_positions: list of (x, y) tuples, one per receiver.
                            Minimum 3 receivers required.
    """

    def __init__(
        self,
        receiver_positions: Optional[list[tuple[float, float]]] = None,
    ):
        if receiver_positions is None:
            receiver_positions = _DEFAULT_RECEIVERS

        if len(receiver_positions) < 3:
            raise ValueError("TDOA requires at least 3 receivers.")

        self.receivers = np.array(receiver_positions, dtype=float)  # (R, 2)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _true_delays(self, transmitter_pos: np.ndarray) -> np.ndarray:
        """
        ⚠ SIMULATED — Compute noiseless one-way propagation delays (seconds)
        from the transmitter to each receiver using Euclidean geometry.

        Returns:
            delays: (R,) array of time-of-arrival in seconds
        """
        diffs    = self.receivers - transmitter_pos   # (R, 2)
        distances = np.linalg.norm(diffs, axis=1)     # (R,)
        return distances / _C

    def _tdoa_from_delays(self, delays: np.ndarray) -> np.ndarray:
        """
        Convert raw arrival times to TDOA values referenced to receiver 0.

        Returns:
            tdoas: (R-1,) array where tdoa[i] = t[i+1] - t[0]
        """
        return delays[1:] - delays[0]

    def _residuals(
        self,
        pos: np.ndarray,
        observed_tdoas: np.ndarray,
    ) -> np.ndarray:
        """
        Residual function for the least-squares solver.

        Returns the difference between the TDOA values predicted for a
        candidate position and the actually observed (noisy) TDOAs.
        """
        predicted_delays = self._true_delays(pos)
        predicted_tdoas  = self._tdoa_from_delays(predicted_delays)
        return predicted_tdoas - observed_tdoas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_detection(
        self,
        true_transmitter_pos: tuple[float, float],
        noise_std_meters: float = 50.0,
    ) -> dict:
        """
        ⚠ SIMULATED — Generate synthetic TDOA measurements for a transmitter.

        Computes the geometrically exact time delays from `true_transmitter_pos`
        to each receiver, then adds independent Gaussian noise whose standard
        deviation corresponds to `noise_std_meters` of ranging error.

        Args:
            true_transmitter_pos : (x, y) ground-truth position in metres
            noise_std_meters     : 1-sigma ranging noise in metres (default 50 m)
                                   Typical values:
                                     10–50 m  → high-quality hardware
                                    100–300 m → software-defined radio
                                   1000 m+   → heavily multipath / urban

        Returns:
            dict with keys:
                "true_pos"          : np.ndarray (2,)
                "true_delays_s"     : np.ndarray (R,)  noiseless delays
                "noisy_delays_s"    : np.ndarray (R,)  noise-corrupted delays
                "tdoas_s"           : np.ndarray (R-1) TDOA values (referenced to rx 0)
                "noise_std_meters"  : float
        """
        tx = np.array(true_transmitter_pos, dtype=float)
        true_delays = self._true_delays(tx)

        # Convert range noise to time noise: σ_t = σ_d / c
        noise_std_seconds = noise_std_meters / _C
        noise = self._rng.normal(0.0, noise_std_seconds, size=len(self.receivers))
        noisy_delays = true_delays + noise

        return {
            "true_pos":         tx,
            "true_delays_s":    true_delays,
            "noisy_delays_s":   noisy_delays,
            "tdoas_s":          self._tdoa_from_delays(noisy_delays),
            "noise_std_meters": noise_std_meters,
        }

    # ------------------------------------------------------------------

    def estimate_position(
        self,
        time_delays: list[float],
    ) -> tuple[float, float, float]:
        """
        ⚠ SIMULATED — Estimate transmitter position from TDOA measurements.

        Solves a non-linear least-squares problem: find (x, y) that minimises
        the sum of squared differences between the observed TDOAs and those
        predicted by the geometry.

        The solver is initialised at the centroid of the receiver array and
        uses the Levenberg-Marquardt method (via scipy.optimize.least_squares).

        Args:
            time_delays : list of R floats — one arrival time (seconds) per
                          receiver, in the same order as receiver_positions.
                          These are the raw TOA values, not TDOAs.

        Returns:
            (estimated_x, estimated_y, confidence_radius_meters)

            confidence_radius_meters is an approximate 1-sigma error circle
            derived from the least-squares residual norm, scaled to metres.
            It reflects the internal consistency of the solution, NOT the
            absolute accuracy (which depends on noise_std and geometry).
        """
        delays = np.array(time_delays, dtype=float)

        if len(delays) != len(self.receivers):
            raise ValueError(
                f"Expected {len(self.receivers)} delays, got {len(delays)}."
            )

        observed_tdoas = self._tdoa_from_delays(delays)

        # Initial guess: centroid of receiver positions
        x0 = self.receivers.mean(axis=0)

        result = least_squares(
            fun=self._residuals,
            x0=x0,
            args=(observed_tdoas,),
            method="lm",          # Levenberg-Marquardt — no bounds, fast
        )

        est_x, est_y = result.x

        # Confidence radius: convert residual norm (seconds) back to metres.
        # This gives an approximate position uncertainty based on how well the
        # solution satisfies the TDOA constraints.
        residual_norm_meters = np.linalg.norm(result.fun) * _C
        n_constraints = len(observed_tdoas)
        confidence_radius = residual_norm_meters * np.sqrt(n_constraints)

        return float(est_x), float(est_y), float(confidence_radius)

    # ------------------------------------------------------------------

    def visualize(
        self,
        true_pos: tuple[float, float],
        estimated_pos: tuple[float, float],
        confidence_radius: float,
        noise_std_meters: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        ⚠ SIMULATED — Plot receiver geometry, true position, and estimated
        position with a confidence circle.

        Args:
            true_pos           : (x, y) ground-truth transmitter position (m)
            estimated_pos      : (x, y) from estimate_position()
            confidence_radius  : radius (m) from estimate_position()
            noise_std_meters   : if provided, shown in the title for context
            save_path          : if set, save figure to this path instead of
                                 calling plt.show()
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

        # --- receivers -------------------------------------------------------
        rx = self.receivers
        ax.scatter(
            rx[:, 0], rx[:, 1],
            marker="^", s=180, color="steelblue", zorder=5, label="Receiver",
        )
        for i, (x, y) in enumerate(rx):
            ax.annotate(
                f" RX{i}", (x, y),
                fontsize=9, color="steelblue", va="bottom",
            )

        # Lines connecting receivers (show the array geometry)
        for i in range(len(rx)):
            for j in range(i + 1, len(rx)):
                ax.plot(
                    [rx[i, 0], rx[j, 0]],
                    [rx[i, 1], rx[j, 1]],
                    color="steelblue", linewidth=0.6, linestyle="--", alpha=0.4,
                )

        # --- true position ---------------------------------------------------
        tx, ty = true_pos
        ax.scatter(
            tx, ty,
            marker="*", s=300, color="crimson", zorder=6, label="True position",
        )

        # --- estimated position + confidence circle --------------------------
        ex, ey = estimated_pos
        ax.scatter(
            ex, ey,
            marker="x", s=160, color="darkorange", linewidths=2.5,
            zorder=6, label=f"Estimated (r={confidence_radius:.0f} m)",
        )
        circle = mpatches.Circle(
            (ex, ey), confidence_radius,
            fill=False, edgecolor="darkorange", linewidth=1.5,
            linestyle=":", label="Confidence circle",
        )
        ax.add_patch(circle)

        # --- error line ------------------------------------------------------
        pos_error = np.linalg.norm(np.array(estimated_pos) - np.array(true_pos))
        ax.plot(
            [tx, ex], [ty, ey],
            color="grey", linewidth=1.2, linestyle="-",
            label=f"Position error: {pos_error:.1f} m",
        )

        # --- labels ----------------------------------------------------------
        noise_tag = f"  |  noise σ={noise_std_meters:.0f} m" if noise_std_meters else ""
        ax.set_title(
            f"SPECTER  ⚠ SIMULATED  —  TDOA Geolocation{noise_tag}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("East  (m)")
        ax.set_ylabel("North (m)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------

    def run_simulation(
        self,
        true_transmitter_pos: tuple[float, float],
        noise_std_meters: float = 50.0,
        visualize: bool = False,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        ⚠ SIMULATED — Convenience end-to-end simulation:
            detect → estimate → (optional) visualize.

        Args:
            true_transmitter_pos : ground-truth (x, y) in metres
            noise_std_meters     : Gaussian ranging noise std (metres)
            visualize            : if True, generate and display/save the plot
            save_path            : plot save path (passed through to visualize())

        Returns:
            dict with keys:
                "true_pos"           : (x, y) tuple
                "estimated_pos"      : (x, y) tuple
                "confidence_radius"  : float (metres)
                "position_error_m"   : float — Euclidean error vs ground truth
                "tdoas_s"            : np.ndarray of TDOA values used
                "noise_std_meters"   : float
        """
        detection = self.simulate_detection(true_transmitter_pos, noise_std_meters)
        ex, ey, radius = self.estimate_position(detection["noisy_delays_s"].tolist())

        true_xy = tuple(detection["true_pos"])
        est_xy  = (ex, ey)
        error_m = float(np.linalg.norm(detection["true_pos"] - np.array(est_xy)))

        if visualize:
            self.visualize(
                true_pos=true_xy,
                estimated_pos=est_xy,
                confidence_radius=radius,
                noise_std_meters=noise_std_meters,
                save_path=save_path,
            )

        return {
            "true_pos":          true_xy,
            "estimated_pos":     est_xy,
            "confidence_radius": radius,
            "position_error_m":  error_m,
            "tdoas_s":           detection["tdoas_s"],
            "noise_std_meters":  noise_std_meters,
        }
