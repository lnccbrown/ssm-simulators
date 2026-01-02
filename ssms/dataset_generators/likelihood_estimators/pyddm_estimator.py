"""PyDDM-based likelihood estimator using Fokker-Planck solutions."""

import numpy as np
import pandas as pd
from typing import Any, Dict
from scipy.interpolate import interp1d


class PyDDMLikelihoodEstimator:
    """Likelihood estimator using PyDDM's analytical Fokker-Planck solutions.

    Unlike KDE estimators which resample from simulations, this estimator
    uses PyDDM's Fokker-Planck solver to compute analytical PDFs and
    interpolates them for fast likelihood evaluation.

    Key advantages:
    - No simulation required (purely analytical)
    - Deterministic (no sampling variability)
    - Fast evaluation via interpolation
    - Exact solutions for compatible models

    Limitations:
    - Only works for PyDDM-compatible models (single-particle, two-choice, Gaussian noise)
    """

    def __init__(
        self,
        pyddm_solution,
        t_domain: np.ndarray,
        interpolation: str = "cubic",
    ):
        """Initialize with a solved PyDDM model.

        The estimator extracts PDFs from the solution internally and
        creates interpolators for fast likelihood evaluation.

        Args:
            pyddm_solution: PyDDM Solution object from model.solve()
            t_domain: Time domain array from model.t_domain()
            interpolation: Interpolation method ('linear' or 'cubic')
        """
        self.solution = pyddm_solution
        self.t_domain = t_domain

        # Extract PDFs internally using modern PyDDM API
        self.pdf_correct = pyddm_solution.pdf("correct")
        self.pdf_error = pyddm_solution.pdf("error")

        # Create interpolators for fast evaluation
        # Use bounds_error=False and fill_value=0 for RTs outside domain
        self._interp_correct = interp1d(
            t_domain,
            self.pdf_correct,
            kind=interpolation,
            bounds_error=False,
            fill_value=0.0,
        )
        self._interp_error = interp1d(
            t_domain,
            self.pdf_error,
            kind=interpolation,
            bounds_error=False,
            fill_value=0.0,
        )

        # Store metadata for protocol compliance
        self._metadata = {
            "max_t": float(t_domain[-1]),
            "dt": float(t_domain[1] - t_domain[0]),
            "possible_choices": [-1, 1],
            "n_choices": 2,
        }

    def fit(self, simulations: Dict[str, Any]) -> None:
        """Fit is a no-op for analytical estimator (already fitted).

        This method exists for protocol compliance with LikelihoodEstimatorProtocol.
        PyDDM solutions are already "fitted" when the Fokker-Planck equation is solved.

        Args:
            simulations: Ignored (not used by analytical estimator)
        """
        pass  # Analytical PDFs don't need fitting

    def evaluate(self, rts: np.ndarray, choices: np.ndarray) -> np.ndarray:
        """Evaluate log-likelihood at (RT, choice) pairs.

        Uses interpolation for fast evaluation. RTs outside the time domain
        are assigned zero probability (log-likelihood = -inf, clipped to -66).

        Args:
            rts: Reaction times, shape (n,)
            choices: Choices in {-1, 1}, shape (n,)

        Returns:
            Log-likelihoods, shape (n,)
        """
        rts = np.asarray(rts).ravel()
        choices = np.asarray(choices).ravel()

        # Interpolate PDFs based on choice
        pdf_values = np.zeros(len(rts))
        correct_mask = choices == 1
        error_mask = choices == -1

        pdf_values[correct_mask] = self._interp_correct(rts[correct_mask])
        pdf_values[error_mask] = self._interp_error(rts[error_mask])

        # Clip to avoid log(0), using same threshold as KDE estimator
        pdf_values = np.clip(pdf_values, 1e-29, None)

        return np.log(pdf_values)

    def sample(
        self, n_samples: int, random_state: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample (RT, choice) pairs from the analytical distribution.

        Uses PyDDM's built-in sampling method for efficient sampling.

        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility. If None, uses non-reproducible random behavior.

        Returns:
            Dictionary with keys:
                - 'rts': Reaction times, shape (n_samples,)
                - 'choices': Choices in {-1, 1}, shape (n_samples,)

        Raises:
            RuntimeError: If unable to collect enough decided trials after
                max_attempts (indicates P(undecided) is too high)

        Note:
            - PyDDM may produce "undecided" trials (process didn't reach boundary
              within T_dur), which appear as NaN. These are filtered out.
            - PyDDM's Sample object segregates trials by boundary (all upper, then
              all lower). We always shuffle to provide randomized samples consistent
              with the rest of ssm-simulators.
            - If P(undecided) is high, multiple sampling rounds may be needed to
              collect enough decided trials.
        """
        # Sample trials (sample exactly what's needed to avoid over-sampling)
        pyddm_sample = self.solution.sample(k=n_samples)

        # Convert to DataFrame and filter out undecided (NaN) trials
        df = pyddm_sample.to_pandas_dataframe().dropna()

        # If we don't have enough decided trials, sample additional batches
        # This is necessary because some trials may be undecided
        max_attempts = 5
        attempt = 0
        while len(df) < n_samples and attempt < max_attempts:
            additional_needed = n_samples - len(df)
            additional_sample = self.solution.sample(k=additional_needed)
            additional_df = additional_sample.to_pandas_dataframe().dropna()
            df = pd.concat([df, additional_df], ignore_index=True)
            attempt += 1

        # Final check
        if len(df) < n_samples:
            p_undecided = self.solution.prob_undecided()
            raise RuntimeError(
                f"Unable to collect {n_samples} decided trials after {attempt + 1} attempts. "
                f"Got {len(df)} decided trials. P(undecided)={p_undecided:.4f} may be too high. "
                f"This should have been caught by max_undecided_prob threshold check."
            )

        # ALWAYS shuffle to remove PyDDM's systematic ordering (samples are segregated by boundary)
        # PyDDM's DataFrame has structure: [undecided as NaN | all upper boundary | all lower boundary]
        # We shuffle to provide a consistent interface with the rest of ssm-simulators
        df = df.sample(
            n=min(n_samples, len(df)), replace=False, random_state=random_state
        ).reset_index(drop=True)

        # Extract data
        all_rts = df["RT"].to_numpy()
        # PyDDM uses choice in {0, 1}, convert to ssms format {-1, 1}
        all_choices = np.where(df["choice"] == 1, 1, -1)

        return {
            "rts": all_rts.ravel(),
            "choices": all_choices.ravel(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the estimator.

        Returns:
            Dictionary containing:
                - max_t: Maximum time in domain
                - dt: Time step
                - possible_choices: List of possible choice values
                - n_choices: Number of choices
        """
        return self._metadata
