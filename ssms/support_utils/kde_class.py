# KDE GENERATORS
from copy import deepcopy

import numpy as np
from sklearn.neighbors import KernelDensity

# from ssms.basic_simulators import simulator


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
        generate_base_kdes(auto_bandwidth=True, bandwidth_type='silverman')
            Generates kdes from rt data.
        kde_eval(data=([], []), log_eval=True)
            Evaluates kde log likelihood at chosen points.
        kde_sample(n_samples=2000, use_empirical_choice_p=True, alternate_choice_p=0)
            Samples from a given kde.
        attach_data_from_simulator(simulator_data={'rts':[0, 2, 4], 'choices':[-1, 1, -1], 'metadata':{}}))
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

        if displace_t:
            # TODO: #84 Replace with exception  # noqa: FIX002
            assert (
                np.unique(simulator_data["metadata"]["t"]).shape[0] == 1
            ), "Multiple t values in simulator data. Can't shift."  # noqa: S101
            self.displace_t_val = np.unique(simulator_data["metadata"]["t"])[0]
            self.displace_t = True
        else:
            self.displace_t = False

        self.attach_data_from_simulator(simulator_data)
        self.generate_base_kdes(
            auto_bandwidth=auto_bandwidth, bandwidth_type=bandwidth_type
        )

    # Function to compute bandwidth parameters given data-set
    # (At this point using Silverman rule)
    def compute_bandwidths(self, bandwidth_type="silverman"):
        """
        Computes bandwidths for each choice from rt data.

        Arguments:
        ----------
        type: string
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.

        Returns:
        --------
        bandwidths: list
            List of bandwidths for each choice.
        """

        # For now allows only silverman rule
        self.bandwidths = []
        if bandwidth_type == "silverman":
            for i in range(0, len(self.data["choices"]), 1):
                if len(self.data["log_rts"][i]) == 0:
                    self.bandwidths.append("no_base_data")
                else:
                    bandwidth_tmp = bandwidth_silverman(
                        sample=(self.data["log_rts"][i])
                    )
                    if bandwidth_tmp > 0:
                        self.bandwidths.append(bandwidth_tmp)
                    else:
                        self.bandwidths.append("no_base_data")

    # Function to generate basic kdes
    # I call the function generate_base_kdes because
    # in the final evaluation computations
    # we adjust the input and output of the kdes
    # appropriately (we do not use them directly)
    def generate_base_kdes(self, auto_bandwidth=True, bandwidth_type="silverman"):
        """
        Generates kdes from rt data. We apply gaussian kernels to the log of the rts.

        Arguments:
        ----------
        auto_bandwidth: boolean
            Whether to compute bandwidths automatically, default is True.
        bandwidth_type: string
            Type of bandwidth to use, default is 'silverman' which follows silverman rule.
        Returns:
        --------
        base_kdes: list
            List of kdes for each choice. (These get attached to the base_kdes attribute of the class, not returned)
        """
        # Compute bandwidth parameters
        if auto_bandwidth:
            self.compute_bandwidths(type=bandwidth_type)

        # Generate the kdes
        self.base_kdes = []
        for i in range(0, len(self.data["choices"]), 1):
            if self.bandwidths[i] == "no_base_data":
                self.base_kdes.append("no_base_data")
            else:
                self.base_kdes.append(
                    KernelDensity(kernel="gaussian", bandwidth=self.bandwidths[i]).fit(
                        np.log(self.data["rts"][i])
                    )
                )

    # Function to evaluate the kde log likelihood at chosen points
    # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B006, FIX002
    def kde_eval(
        self, data={}, log_eval=True, lb=-66.774, eps=10e-5, filter_rts=-999
    ):  # noqa: B006
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

        if "log_rts" in data.keys() and ("rts" not in data.keys()):
            data_internal["log_rts"] = np.expand_dims(
                data["log_rts"][data["log_rts"] != filter_rts], axis=1
            )
            data_internal["choices"] = np.expand_dims(
                data["choices"][data["log_rts"] != filter_rts], axis=1
            )
            assert (
                data["log_rts"].shape == data["choices"].shape
            ), "rts and choices need to have matching shapes in data dictionary!"  # noqa: S101  # TODO: #84 Replace with exception  # noqa: FIX002
            return self._kde_eval_log_rt(
                data=data_internal, log_eval=log_eval, lb=lb, eps=eps
            )
        elif "rts" in data.keys() and ("log_rts" not in data.keys()):
            data_internal["rts"] = np.expand_dims(
                data["rts"][data["rts"] != filter_rts], axis=1
            )
            data_internal["choices"] = np.expand_dims(
                data["choices"][data["rts"] != filter_rts], axis=1
            )
            assert (
                data_internal["rts"].shape == data_internal["choices"].shape
            ), "rts and choices need to have matching shapes in data dictionary!"  # noqa: S101  # TODO: #84 Replace with exception  # noqa: FIX002
            return self._kde_eval_(
                data=data_internal, log_eval=log_eval, lb=lb, eps=eps
            )
        elif ("log_rts" in data.keys()) and ("rts" in data.keys()):
            data_internal["rts"] = np.expand_dims(
                data["rts"][data["rts"] != filter_rts], axis=1
            )
            data_internal["choices"] = np.expand_dims(
                data["choices"][data["rts"] != filter_rts], axis=1
            )
            assert (
                data_internal["rts"].shape == data_internal["choices"].shape
            ), "rts and choices need to have matching shapes in data dictionary!"  # noqa: S101  # TODO: #84 Replace with exception  # noqa: FIX002
            return self._kde_eval_(
                data=data_internal, log_eval=log_eval, lb=lb, eps=eps
            )
        else:
            raise ValueError(
                "data dictionary must contain either rts or log_rts as keys!"
            )

    # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B006, FIX002
    def _kde_eval_(
        self,
        data={},  # noqa: B006
        log_eval=True,
        lb=-66.774,
        eps=10e-5,
    ):  # kde
        """
        Evaluates kde log likelihood at chosen points.

        Arguments:
        ----------
        data: tuple
            Tuple of (rts, choices) to evaluate the kde at.
        log_eval: boolean
            Whether to return log likelihood or likelihood, default is True.

        Returns:
        --------
        log_kde_eval: array
            Array of log likelihoods for each (rt, choice) pair.
        """

        # Initializations
        if self.displace_t is True:
            displaced_rts = data["rts"] - self.displace_t_val
            log_rts = np.log(np.maximum(displaced_rts, eps))
        else:
            displaced_rts = data["rts"]
            log_rts = np.log(data["rts"])

        log_kde_eval = np.zeros(data["choices"].shape)  # np.log(data['rts'])
        choices = np.unique(data["choices"])

        # Main loop
        for c in choices:
            # Get data indices where choice == c
            choice_idx_tmp = np.where(data["choices"] == c)

            # Main step: Evaluate likelihood for rts corresponding
            # to choice == c
            if self.base_kdes[self.data["choices"].index(c)] == "no_base_data":
                log_kde_eval[choice_idx_tmp] = np.log(
                    1 / self.data["n_trials"]
                ) + np.log(
                    1 / self.simulator_info["max_t"]
                )  # -66.77497 # the number corresponds to log(1e-29)
                # --> should rather be log(1 / n) + log(1 / 20)
            else:
                log_kde_eval_out_tmp = log_rts[choice_idx_tmp]
                log_kde_eval_out_tmp[displaced_rts[choice_idx_tmp] <= 0] = lb
                log_kde_eval_out_tmp[displaced_rts[choice_idx_tmp] > 0] = (
                    np.log(
                        self.data["choice_proportions"][self.data["choices"].index(c)]
                    )
                    + self.base_kdes[self.data["choices"].index(c)].score_samples(
                        np.expand_dims(
                            log_rts[choice_idx_tmp][displaced_rts[choice_idx_tmp] > 0],
                            1,
                        )
                    )
                    - log_rts[choice_idx_tmp][displaced_rts[choice_idx_tmp] > 0]
                )
                log_kde_eval[choice_idx_tmp] = log_kde_eval_out_tmp

        if log_eval is True:
            return np.squeeze(log_kde_eval)
        else:
            return np.squeeze(np.exp(log_kde_eval))

    # Function to evaluate the kde log likelihood at chosen points
    def _kde_eval_log_rt(
        self,
        data={},  # noqa: B006
        log_eval=True,
        lb=-66.774,
        eps=10e-5,
    ):  # kde # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B006, FIX002
        """
        Evaluates kde log likelihood at chosen points.

        Arguments:
        ----------
        data: tuple
            Tuple of (rts, choices) to evaluate the kde at.
        log_eval: boolean
            Whether to return log likelihood or likelihood, default is True.

        Returns:
        --------
        log_kde_eval: array
            Array of log likelihoods for each (rt, choice) pair.
        """

        if self.displace_t is True:
            displaced_rts = np.exp(data["log_rts"]) - self.displace_t_val
            log_rts = np.log(np.maximum(displaced_rts, eps))
        else:
            displaced_rts = np.exp(data["log_rts"])
            log_rts = np.log(np.maximum(np.exp(data["log_rts"], eps)))

        log_kde_eval = np.zeros(data["choices"].shape)  # np.log(data['rts'])
        choices = np.unique(data["choices"])

        # Main loop
        for c in choices:
            # Get data indices where choice == c
            choice_idx_tmp = np.where(data["choices"] == c)

            # Main step: Evaluate likelihood for rts corresponding
            # to choice == c
            if self.base_kdes[self.data["choices"].index(c)] == "no_base_data":
                # AF-TODO: Check if it make sense to apply the np.log(1 / max_t)
                # here, since if we operate on log rts, log rts are not uniform
                # on the interval [log(~0), log(max_t)]
                log_kde_eval[choice_idx_tmp] = np.log(
                    1 / self.data["n_trials"]
                ) + np.log(
                    1 / self.simulator_info["max_t"]
                )  # -66.77497 # the number corresponds to log(1e-29)
                # --> should rather be log(1 / n) + log(1 / 20)
            else:
                log_kde_eval_out_tmp = log_rts[choice_idx_tmp]
                log_kde_eval_out_tmp[displaced_rts[choice_idx_tmp] <= 0] = lb

                log_kde_eval_out_tmp[displaced_rts[choice_idx_tmp] > 0] = np.log(
                    self.data["choice_proportions"][self.data["choices"].index(c)]
                ) + self.base_kdes[self.data["choices"].index(c)].score_samples(
                    np.expand_dims(
                        log_rts[choice_idx_tmp][displaced_rts[choice_idx_tmp] > 0], 1
                    )
                )
                log_kde_eval[choice_idx_tmp] = log_kde_eval_out_tmp

        if log_eval is True:
            return np.squeeze(log_kde_eval)
        else:
            return np.squeeze(np.exp(log_kde_eval))

    def kde_sample(
        self, n_samples=2000, use_empirical_choice_p=True, alternate_choice_p=0
    ):
        """
        Samples from a given kde.

        Arguments:
        ----------
        n_samples: int
            Number of samples to draw.
        use_empirical_choice_p: boolean
            Whether to use empirical choice proportions, default is True. (Note 'empirical' here,
            refers to the originally attached datasets that served as the basis to generate the choice-wise
            kdes)
        alternate_choice_p: array
            Array of choice proportions to use, default is 0. (Note 'alternate' here refers to 'alternative'
            to the 'empirical' choice proportions)

        """

        # sorting the which list in ascending order
        # this implies that we return the kde_samples array so that the
        # indices reflect 'choice-labels' as provided in 'which' in ascending order
        # kde_samples = []

        rts = np.zeros((n_samples, 1))
        choices = np.zeros((n_samples, 1))

        n_by_choice = []
        for i in range(0, len(self.data["choices"]), 1):
            if use_empirical_choice_p is True:
                n_by_choice.append(
                    round(n_samples * self.data["choice_proportions"][i])
                )
            else:
                n_by_choice.append(round(n_samples * alternate_choice_p[i]))

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

        if self.displace_t is True:
            rts = rts + self.displace_t_val
        return {
            "rts": rts,
            "log_rts": np.log(rts),
            "choices": choices,
            "metadata": self.simulator_info,
        }

    # Helper function to transform ddm simulator output to dataset suitable for
    # the kde function class
    def attach_data_from_simulator(
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
    sample=[  # noqa: B006
        0,
        0,
        0,
    ],  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B006, B008, FIX002
    std_cutoff=1e-3,
    std_proc="restrict",  # options 'kill', 'restrict'
    std_n_1=10,  # HERE WE CAN ALLOW FOR SOMETHING MORE INTELLIGENT
):
    """
    Computes silverman bandwidth for an array of samples (rts in our context, but general).

    Arguments:
    ----------
    sample: array
        Array of samples to compute bandwidth for.
    std_cutoff: float
        Cutoff for std, default is 1e-3.
        (If sample-std is smaller than this, we either kill it or restrict it to this value)
    std_proc: string
        How to deal with small stds, default is 'restrict'. (Options: 'kill', 'restrict')
    std_n_1: float
        Value to use if n = 1, default is 10. (Not clear if the default is sensible here)

    Returns:
    --------
    bandwidth: float
        Silverman bandwidth for the given sample. This is applied as the bandwidth parameter
        when generating gaussian-based kdes in the LogKDE class.
    """
    # Compute number of samples
    n = len(sample)

    # Deal with very small stds and n = 1 case
    if n > 1:
        # Compute std of sample
        std = np.std(sample)
        if std < std_cutoff:
            if std_proc == "restrict":
                std = std_cutoff
            if std_proc == "kill":
                std = 0
    else:
        # AF-Comment: This is a bit of a weakness (can be arbitrarily incorrect)
        std = std_n_1

    return np.power((4 / 3), 1 / 5) * std * np.power(n, (-1 / 5))
