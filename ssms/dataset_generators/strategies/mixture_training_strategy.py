"""Resample mixture training data generation strategy.

This strategy generates training data by mixing three types of samples:
1. Samples drawn from the likelihood estimator (KDE or analytical PDF)
2. Uniform samples in positive RT space (to encourage exploration)
3. Uniform samples in negative RT space (to learn the hard boundary at t=0)

This approach helps the network learn both the likelihood surface and its boundaries.
"""

from typing import Any, Dict

import numpy as np


class MixtureTrainingStrategy:
    """Training data strategy using mixture of resamples and uniform draws.

    This strategy generates training data by:
    1. Resampling (RT, choice) pairs from the fitted likelihood estimator
    2. Generating uniform samples in positive RT space [0, max_t]
    3. Generating uniform samples in negative RT space [-1, 0]

    For each sample, the strategy evaluates the log-likelihood and combines:
    - Parameters (theta)
    - RT and choice
    - Log-likelihood at (RT, choice)

    Attributes
    ----------
    generator_config : dict
        Configuration for data generation (sample counts, mixture probabilities, etc.)
    model_config : dict
        Model configuration (params, nchoices, etc.)

    Examples
    --------
    >>> strategy = MixtureTrainingStrategy(generator_config, model_config)
    >>> training_data = strategy.generate(theta, likelihood_estimator)
    >>> assert training_data.shape == (n_training_samples, n_features)
    """

    def __init__(self, generator_config: dict, model_config: dict):
        """Initialize mixture training strategy.

        Arguments
        ---------
        generator_config : dict
            Configuration dictionary containing:
            - 'n_training_samples_by_parameter_set': Total samples to generate
            - 'data_mixture_probabilities': [p_estimator, p_unif_up, p_unif_down]
            - 'separate_response_channels': Whether to one-hot encode choices
            - 'negative_rt_log_likelihood': Log-likelihood value for negative RTs
        model_config : dict
            Model configuration containing:
            - 'params': List of parameter names
            - 'nchoices': Number of possible choices
        """
        self.generator_config = generator_config
        self.model_config = model_config

    def generate(
        self,
        theta: Dict[str, Any],
        likelihood_estimator,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate training data using mixture strategy.

        Arguments
        ---------
        theta : dict
            Parameter dictionary (e.g., {'v': 1.0, 'a': 2.0, ...})
        likelihood_estimator : LikelihoodEstimatorProtocol
            Fitted likelihood estimator (must support sample() and evaluate())
        random_state : int | None, optional
            Random seed for reproducibility. If None, uses non-reproducible random behavior.
            Controls all random operations including sampling from estimator and shuffling output.

        Returns
        -------
        np.ndarray
            Training data array of shape (n_samples, n_features) where:
            - First columns: theta parameters
            - Middle columns: RT and choice (possibly one-hot encoded)
            - Last column: log-likelihood

            Note: Rows are shuffled to avoid ordering bias in training batches.

        Raises
        ------
        ValueError
            If mixture probabilities don't lead to valid sample counts

        Notes
        -----
        Extracted from lan_mlp.py lines 268-360
        """
        # Create seeded random generator for reproducibility
        rng = np.random.default_rng(random_state)
        # Extract configuration from nested structure
        n = self.generator_config["training"]["n_samples_per_param"]
        p = self.generator_config["training"]["mixture_probabilities"]

        # Calculate sample counts for each component
        n_kde = int(n * p[0])
        n_unif_up = int(n * p[1])
        n_unif_down = int(n * p[2])

        # Adjust for rounding errors
        total = n_kde + n_unif_up + n_unif_down
        if total != n:
            n_kde += n - total
            if n_kde < 0:
                raise ValueError(
                    f"Rounding error too large: cannot adjust n_kde to {n_kde}. "
                    f"n={n}, p={p}, "
                    f"n_kde={n_kde - (n - total)}, "
                    f"n_unif_up={n_unif_up}, "
                    f"n_unif_down={n_unif_down}"
                )

        # Get metadata from estimator
        metadata = likelihood_estimator.get_metadata()

        # Allocate output array
        separate_channels = self.generator_config.get("training", {}).get(
            "separate_response_channels", False
        )
        if separate_channels:
            n_features = 2 + self.model_config["nchoices"] + len(theta.items())
        else:
            n_features = 3 + len(theta.items())

        out = np.zeros((n_kde + n_unif_up + n_unif_down, n_features))

        # Fill in theta parameters (broadcast to all rows)
        # Handle both scalar and array theta values
        theta_values = []
        for key_ in self.model_config["params"]:
            val = theta[key_]
            # Convert scalars to arrays
            if np.isscalar(val):
                theta_values.append(val)
            else:
                # If already an array, take the first element
                theta_values.append(np.asarray(val).flat[0])

        theta_array = np.array([theta_values])  # Shape: (1, n_params)
        out[:, : len(theta.items())] = np.tile(
            theta_array, (n_kde + n_unif_up + n_unif_down, 1)
        )

        # === 1. KDE/Estimator samples ===
        samples_kde = likelihood_estimator.sample(
            n_samples=n_kde, random_state=random_state
        )
        likelihoods_kde = likelihood_estimator.evaluate(
            samples_kde["rts"], samples_kde["choices"]
        )

        if separate_channels:
            # One-hot encode choices
            out[:n_kde, (-2 - self.model_config["nchoices"])] = samples_kde["rts"]

            r_cnt = 0
            choices = samples_kde["choices"]
            for response in metadata["possible_choices"]:
                out[:n_kde, ((-1 - self.model_config["nchoices"]) + r_cnt)] = (
                    choices == response
                ).astype(int)
                r_cnt += 1
        else:
            out[:n_kde, -3] = samples_kde["rts"]
            out[:n_kde, -2] = samples_kde["choices"]

        out[:n_kde, -1] = likelihoods_kde

        # === 2. Positive uniform samples ===
        choice_tmp = rng.choice(metadata["possible_choices"], size=n_unif_up)

        if metadata["max_t"] < 100:
            rt_tmp = rng.uniform(low=0.0001, high=metadata["max_t"], size=n_unif_up)
        else:
            rt_tmp = rng.uniform(low=0.0001, high=100, size=n_unif_up)

        likelihoods_unif = likelihood_estimator.evaluate(rt_tmp, choice_tmp)

        # Note: Uniform samples always use -3/-2 indexing, even with separate_response_channels
        # This matches the original lan_mlp.py behavior (lines 346-348)
        # AF-TODO: Check if we can do something better here, to allow separate channels throughout!
        out[n_kde : (n_kde + n_unif_up), -3] = rt_tmp
        out[n_kde : (n_kde + n_unif_up), -2] = choice_tmp
        out[n_kde : (n_kde + n_unif_up), -1] = likelihoods_unif

        # === 3. Negative uniform samples ===
        choice_tmp = rng.choice(metadata["possible_choices"], size=n_unif_down)

        rt_tmp = rng.uniform(low=-1.0, high=0.0001, size=n_unif_down)

        out[(n_kde + n_unif_up) :, -3] = rt_tmp
        out[(n_kde + n_unif_up) :, -2] = choice_tmp
        out[(n_kde + n_unif_up) :, -1] = self.generator_config.get("training", {}).get(
            "negative_rt_log_likelihood", -66.77497
        )

        # Shuffle rows to avoid ordering bias in training batches
        rng.shuffle(out)

        return out.astype(np.float32)
