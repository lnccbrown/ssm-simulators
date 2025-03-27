# KDE GENERATORS
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from sklearn.neighbors import KernelDensity

"""
    This module contains a class for generating kdes from data.
"""


class LogKDE:
    """
    Class for generating kdes from (rt, choice) data. Works for any number of choices.

    Attributes
    ----------
        simulator_data: dict, default<None
            Dictionary of the type {'rts':[], 'choices':[], 'metadata':{}}.
            Follows the format of simulator returns in this package.
        bandwidth_type: string
            type of bandwidth to use, default is 'silverman'
        auto_bandwidth: boolean
            whether to compute bandwidths automatically, default is True

    Methods
    -------
        compute_bandwidths(type='silverman')
            Computes bandwidths for each choice from rt data.
        _generate_base_kdes(auto_bandwidth=True, bandwidth_type='silverman')
            Generates kdes from rt data.
        kde_eval(data=([], []), log_eval=True)
            Evaluates kde log likelihood at chosen points.
        kde_sample(n_samples=2000, use_empirical_choice_p=True, alternate_choice_p=0)
            Samples from a given kde.
        _attach_data_from_simulator(simulator_data={'rts':[0, 2, 4], 'choices':[-1, 1, -1], 'metadata':{}}))
            Helper function to transform ddm simulator output
            to dataset suitable for the kde function class.

    Returns:
        _type_: _description_
    """

    # Initialize the class
    def __init__(
        self,
        simulator_data: dict,  # as returned by simulator function
        bandwidth_type: str = "silverman",
        auto_bandwidth: bool = True,
        displace_t: bool = False,
    ):
        """Initialize LogKDE class.

        Arguments:
        ----------
            simulator_data: Dictionary containing simulation data with keys 'rts', 'choices', and 'metadata'.
                Follows the format returned by simulator functions in this package.
            bandwidth_type: Type of bandwidth to use for KDE. Currently only 'silverman' is supported.
                Defaults to 'silverman'.
            auto_bandwidth: Whether to automatically compute bandwidths based on the data.
                If False, bandwidths must be set manually. Defaults to True.
            displace_t: Whether to shift RTs by the t parameter from metadata.
                Only works if all trials have the same t value. Defaults to False.

        Raises:
        -------
            AssertionError: If displace_t is True but metadata contains multiple t values.
        """
        self.simulator_info = simulator_data["metadata"]
        self.displace_t: bool = displace_t

        if self.displace_t:
            t_vals = np.unique(simulator_data["metadata"]["t"])
            if t_vals.shape[0] != 1:
                raise ValueError("Multiple t values in simulator data. Can't shift.")
            self.displace_t_val: float = t_vals[0]

        self._attach_data_from_simulator(simulator_data)
        self._generate_base_kdes(
            auto_bandwidth=auto_bandwidth, bandwidth_type=bandwidth_type
        )

        self.auto_bandwidth: bool = auto_bandwidth
        self.bandwidth_type: str = bandwidth_type

    def compute_bandwidths(
        self,
        bandwidth_type: str = "silverman",
    ) -> list[float | str]:
        """
        Computes bandwidths for each choice from rt data.

        Arguments:
        ----------
        bandwidth_type: str
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.
        return_result: bool
            Whether to return the result. Defaults to False.

        Returns:
        --------
        bandwidths: list[float] | None
            List of bandwidths for each choice if return_result is True, otherwise None.
        """

        # For now allows only silverman rule
        if bandwidth_type == "silverman":
            bandwidths_ = [
                self._compute_bandwidth_for_choice(log_rts)
                for log_rts in self.data["log_rts"]
            ]
        else:
            raise ValueError(f"Bandwidth type {bandwidth_type} not supported yet")
        return bandwidths_

    def _compute_bandwidth_for_choice(self, log_rts):
        if len(log_rts) == 0:
            return "no_base_data"
        else:
            bw_tmp = bandwidth_silverman(sample=log_rts)
            if bw_tmp > 0:
                return bw_tmp
            return "no_base_data"

    def _generate_base_kdes(
        self, auto_bandwidth: bool = True, bandwidth_type: str = "silverman"
    ) -> None:
        """
        Generates kdes from rt data. We apply gaussian kernels to the log of the rts.

        Arguments:
        ----------
        auto_bandwidth: bool
            Whether to compute bandwidths automatically, default is True.
        bandwidth_type: str
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.
        Returns:
        --------
        base_kdes: list
            List of kdes for each choice. (These get attached to the base_kdes attribute of the class, not returned)
        """
        # Compute bandwidth parameters
        if auto_bandwidth:
            self.bandwidths = self.compute_bandwidths(bandwidth_type=bandwidth_type)

        # Generate the kdes
        self.base_kdes = [
            (
                "no_base_data"
                if self.bandwidths[i] == "no_base_data"
                else KernelDensity(kernel="gaussian", bandwidth=self.bandwidths[i]).fit(
                    np.log(self.data["rts"][i])
                )
            )
            for i in range(len(self.data["choices"]))
        ]

    def kde_eval(
        self,
        data: dict,
        log_eval: bool = True,
        lb: float = -66.774,
        eps: float = 10e-5,
        filter_rts: float = -999,
    ) -> np.ndarray:
        """
        Evaluates kde log likelihood at chosen points.

        Arguments:
        ----------
        data: dict
            Dictionary with keys 'rts', and/or 'log_rts' and 'choices' to evaluate the kde at.
            If 'rts' is provided, 'log_rts' is ignored.
        log_eval: boolean
            Whether to return log likelihood or likelihood, default is True.
        lb: float
            Lower bound for log likelihoods, default is -66.774. (This is the log of 1e-29)
        eps: float
            Epsilon value to use for lower bounds on rts.
        filter_rts: float
            Value to filter rts by, default is -999. -999 is the number returned by the
            simulators if we breach max_t or deadline.

        Returns:
        --------
        log_kde_eval: array
            Array of log likelihoods for each (rt, choice) pair.
        """
        # Initializations
        data_internal = deepcopy(data)

        if "log_rts" in data and ("rts" not in data):
            log_rts = data["log_rts"]
            mask = log_rts != filter_rts
            log_rts_filtered = log_rts[mask]
            log_rts_expanded = np.expand_dims(log_rts_filtered, axis=1)
            data_internal["rts"] = np.exp(log_rts_expanded)
        elif "rts" in data:
            rts = data["rts"]
            mask = rts != filter_rts
            rts_filtered = rts[mask]
            rts_expanded = np.expand_dims(rts_filtered, axis=1)
            data_internal["rts"] = rts_expanded
        else:
            raise ValueError(
                "data dictionary must contain either rts or log_rts as keys!"
            )

        choices_filtered = data["choices"][mask]
        choices_expanded = np.expand_dims(choices_filtered, axis=1)
        data_internal["choices"] = choices_expanded

        if data_internal["rts"].shape != data_internal["choices"].shape:
            raise ValueError(
                "rts and choices need to have matching shapes in data dictionary!"
            )

        return self.__kde_eval_(data=data_internal, log_eval=log_eval, lb=lb, eps=eps)

    def __kde_eval_(
        self,
        data: dict,  # noqa: B006
        log_eval: bool = True,
        lb: float = -66.774,
        eps: float = 10e-5,
    ) -> np.ndarray:  # kde
        """
        Evaluates kde log likelihood at chosen points.

        Arguments:
        ----------
        data: dict
            Dictionary with keys 'rts', and 'choices' to evaluate the kde at.
        log_eval: bool
            Whether to return log likelihood or likelihood, default is True.
        lb: float
            Lower bound for log likelihoods, default is -66.774.
        eps: float
            Epsilon value to use for lower bounds on rts, default is 10e-5.

        Returns:
        --------
        np.ndarray
            Array of log likelihoods for each (rt, choice) pair if log_eval is True,
            otherwise array of likelihoods.
        """

        # Initializations
        if self.displace_t is True:
            displaced_rts = data["rts"] - self.displace_t_val
        else:
            displaced_rts = data["rts"]

        # The line below is to avoid log(0) and to ensure that the log_rts are always defined
        # however wherever displaced_rts <= 0, we will not evaluate the log_rts directly
        # in the following
        log_rts = np.log(np.maximum(displaced_rts, eps))

        log_kde_eval = np.zeros(data["choices"].shape)
        choices = np.unique(data["choices"])

        # Main loop
        for c in choices:
            choice_idx_tmp = np.where(data["choices"] == c)

            if self.base_kdes[self.data["choices"].index(c)] == "no_base_data":
                # Evaluate likelihood for "choices" for which we have no base data
                # log(1 / n_trials_simulator) + log(1 / max_t)
                log_kde_eval[choice_idx_tmp] = -np.log(
                    self.data["n_trials"] * self.simulator_info["max_t"]
                )
            else:
                # Evaluate likelihood for "choices" for which we have base data
                log_kde_eval_out_tmp = np.zeros_like(log_rts[choice_idx_tmp])
                # Evaluate likelihood explicitly where displaced_rts > 0
                rt_pos_idx = displaced_rts[choice_idx_tmp] > 0
                valid_log_rts_for_choice = log_rts[choice_idx_tmp][rt_pos_idx]
                log_kde_eval_out_tmp[rt_pos_idx] = (
                    np.log(
                        self.data["choice_proportions"][self.data["choices"].index(c)]
                    )
                    + self.base_kdes[self.data["choices"].index(c)].score_samples(
                        np.expand_dims(valid_log_rts_for_choice, 1)
                    )
                    - valid_log_rts_for_choice
                )

                # Apply lower bounds where displaced_rts <= 0 and where original evaluation
                # of likelihood was below the lower bound lb
                log_kde_eval_out_tmp[
                    (log_kde_eval_out_tmp <= lb) | (displaced_rts[choice_idx_tmp] <= 0)
                ] = lb

                log_kde_eval[choice_idx_tmp] = log_kde_eval_out_tmp

        if log_eval:
            return np.squeeze(log_kde_eval)
        return np.squeeze(np.exp(log_kde_eval))

    def kde_sample(
        self,
        n_samples: int = 2000,
        use_empirical_choice_p: bool = True,
        alternate_choice_p: np.ndarray | float = 0.0,
    ) -> dict[str, np.ndarray | dict]:
        """
        Samples from a given kde.

        Arguments:
        ----------
        n_samples: int
            Number of samples to draw.
        use_empirical_choice_p: bool
            Whether to use empirical choice proportions, default is True. (Note 'empirical' here,
            refers to the originally attached datasets that served as the basis to generate the choice-wise
            kdes)
        alternate_choice_p: np.ndarray | float
            Array of choice proportions to use, default is 0. (Note 'alternate' here refers to 'alternative'
            to the 'empirical' choice proportions)

        Returns:
        --------
        dict[str, np.ndarray | dict]
            Dictionary containing:
            - 'rts': np.ndarray - Response times
            - 'log_rts': np.ndarray - Log of response times
            - 'choices': np.ndarray - Choices made
            - 'metadata': dict - Simulator information
        """

        rts = np.zeros((n_samples, 1))
        choices = np.zeros((n_samples, 1))
        if isinstance(alternate_choice_p, float):
            alternate_choice_p = [alternate_choice_p]

        if (
            not len(alternate_choice_p) == len(self.data["choices"])
            and not use_empirical_choice_p
        ):
            raise ValueError(
                "alternate_choice_p must be of the same length as the number of choices"
            )

        n_by_choice = [
            (
                round(n_samples * self.data["choice_proportions"][i])
                if use_empirical_choice_p
                else round(n_samples * alternate_choice_p[i])
            )
            for i in range(len(self.data["choices"]))
        ]

        # Catch a potential dimension error if we ended up rounding up twice
        if sum(n_by_choice) > n_samples:
            n_by_choice[np.argmax(n_by_choice)] -= 1
        elif sum(n_by_choice) < n_samples:
            n_by_choice[np.argmax(n_by_choice)] += 1
            choices[n_samples - 1, 0] = np.random.choice(self.data["choices"])

        # Get samples
        cnt_low = 0
        for i in range(0, len(self.data["choices"]), 1):
            if n_by_choice[i] > 0:
                cnt_high = cnt_low + n_by_choice[i]

                if self.base_kdes[i] != "no_base_data":
                    rts[cnt_low:cnt_high] = np.exp(
                        self.base_kdes[i].sample(n_samples=n_by_choice[i])
                    )
                else:
                    rts[cnt_low:cnt_high, 0] = np.random.uniform(
                        low=0, high=self.simulator_info["max_t"], size=n_by_choice[i]
                    )

                choices[cnt_low:cnt_high, 0] = np.repeat(
                    self.data["choices"][i], n_by_choice[i]
                )
                cnt_low = cnt_high

        if self.displace_t:
            rts = rts + self.displace_t_val

        return {
            "rts": rts,
            "log_rts": np.log(rts),
            "choices": choices,
            "metadata": self.simulator_info,
        }

    # Helper function to transform ddm simulator output to dataset suitable for
    # the kde function class
    def _attach_data_from_simulator(
        self, simulator_data=([0, 2, 4], [-1, 1, -1]), filter_rts=-999
    ):
        """
        Helper function to transform ddm simulator output to dataset suitable for
        the kde function class.

        Arguments:
        ----------
        simulator_data: tuple
            Tuple of (rts, choices, simulator_info) as returned by simulator function.
        filter_rts: float
            Value to filter rts by, default is -999. -999 is the number returned by the
            simulators if we breach max_t or deadline.
        """

        simulator_data = deepcopy(simulator_data)
        choices = np.unique(simulator_data["metadata"]["possible_choices"])
        n = len(simulator_data["choices"])
        self.data = {"rts": [], "log_rts": [], "choices": [], "choice_proportions": []}

        # Loop through the choices made to get proportions and separated out rts
        if "log_rts" in simulator_data.keys() and ("rts" not in simulator_data.keys()):
            simulator_data["rts"] = (
                np.ones(simulator_data["log_rts"].shape) * filter_rts
            )
            simulator_data["rts"][simulator_data["log_rts"] != filter_rts] = np.exp(
                simulator_data["log_rts"][simulator_data["log_rts"] != filter_rts]
            )
        elif "rts" in simulator_data.keys() and (
            "log_rts" not in simulator_data.keys()
        ):
            simulator_data["log_rts"] = (
                np.ones(simulator_data["rts"].shape) * filter_rts
            )
            simulator_data["log_rts"][simulator_data["rts"] != filter_rts] = np.log(
                simulator_data["rts"][simulator_data["rts"] != filter_rts]
            )
        else:
            raise ValueError(
                "simulator_data dictionary must contain either "
                + "rts or log_rts or both as keys!"
            )

        for c in choices:
            rts_tmp = simulator_data["rts"][simulator_data["choices"] == c]
            log_rts_tmp = simulator_data["log_rts"][simulator_data["choices"] == c]

            if self.displace_t is True:
                rts_tmp[rts_tmp != filter_rts] = (
                    rts_tmp[rts_tmp != filter_rts] - self.displace_t_val
                )
                log_rts_tmp[log_rts_tmp != filter_rts] = np.log(
                    np.exp(log_rts_tmp[log_rts_tmp != filter_rts]) - self.displace_t_val
                )

            prop_tmp = len(rts_tmp) / n
            self.data["choices"].append(c)

            self.data["log_rts"].append(
                np.expand_dims(log_rts_tmp[log_rts_tmp != filter_rts], axis=1)
            )
            self.data["rts"].append(
                np.expand_dims(rts_tmp[rts_tmp != filter_rts], axis=1)
            )
            self.data["choice_proportions"].append(prop_tmp)

        self.data["n_trials"] = simulator_data["choices"].shape[0]


# Support functions (accessible from outside the main class defined in script)
def bandwidth_silverman(
    sample: Iterable[float] = (0, 0, 0),
    std_cutoff: float = 1e-3,
    std_proc: str = "restrict",  # options 'kill', 'restrict'
    std_n_1: float = 10.0,  # HERE WE CAN ALLOW FOR SOMETHING MORE INTELLIGENT
) -> np.float64:
    """
    Computes silverman bandwidth for an array of samples (rts in our context, but general).

    Arguments:
    ----------
    sample: np.ndarray
        Array of samples to compute bandwidth for.
    std_cutoff: float
        Cutoff for std, default is 1e-3.
        (If sample-std is smaller than this, we either kill it or restrict it to this value)
    std_proc: str
        How to deal with small stds, default is 'restrict'. (Options: 'kill', 'restrict')
    std_n_1: float
        Value to use if n = 1, default is 10.0. (Not clear if the default is sensible here)

    Returns:
    --------
    float
        Silverman bandwidth for the given sample. This is applied as the bandwidth parameter
        when generating gaussian-based kdes in the LogKDE class.
    """
    # Compute number of samples
    sample_ = np.array(sample)
    n = len(sample_)
    if n == 0:
        raise ValueError("Sample is empty")

    # Deal with very small stds and n = 1 case
    if n > 1:
        std = np.std(sample_)
        # If std is too small,
        # either kill it of restrict to std_cutoff
        if std < std_cutoff:
            std = std_cutoff if std_proc == "restrict" else 0.0
    else:
        # AF-Comment: This is a bit of a weakness (can be arbitrarily incorrect)
        std = std_n_1

    return np.power((4 / (3 * n)), 1 / 5) * std
