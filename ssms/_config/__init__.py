"""Model configuration module for SSM simulators."""

from .ddm import get_ddm_config
# from .ddm_random import (
#     get_ddm_st_config,
#     get_ddm_truncnormt_config,
#     get_ddm_rayleight_config,
#     get_ddm_sdv_config,
# )
# from .ddm_par2 import (
#     get_ddm_par2_config,
#     get_ddm_par2_no_bias_config,
#     get_ddm_par2_conflict_gamma_no_bias_config,
#     get_ddm_par2_angle_no_bias_config,
#     get_ddm_par2_weibull_no_bias_config,
# )
# from .ddm_seq2 import (
#     get_ddm_seq2_config,
#     get_ddm_seq2_no_bias_config,
#     get_ddm_seq2_conflict_gamma_no_bias_config,
#     get_ddm_seq2_angle_no_bias_config,
#     get_ddm_seq2_weibull_no_bias_config,
# )
# from .mic2 import (
#     get_ddm_mic2_adj_config,
#     get_ddm_mic2_adj_no_bias_config,
#     get_ddm_mic2_adj_conflict_gamma_no_bias_config,
#     get_ddm_mic2_adj_angle_no_bias_config,
#     get_ddm_mic2_adj_weibull_no_bias_config,
#     get_ddm_mic2_ornstein_config,
#     get_ddm_mic2_ornstein_no_bias_config,
#     get_ddm_mic2_ornstein_conflict_gamma_no_bias_config,
#     get_ddm_mic2_ornstein_angle_no_bias_config,
#     get_ddm_mic2_ornstein_weibull_no_bias_config,
#     get_ddm_mic2_leak_config,
#     get_ddm_mic2_leak_no_bias_config,
#     get_ddm_mic2_leak_conflict_gamma_no_bias_config,
#     get_ddm_mic2_leak_angle_no_bias_config,
#     get_ddm_mic2_leak_weibull_no_bias_config,
# )
# from .mic2.multinoise import (
#     get_ddm_mic2_multinoise_no_bias_config,
#     get_ddm_mic2_multinoise_conflict_gamma_no_bias_config,
#     get_ddm_mic2_multinoise_angle_no_bias_config,
#     get_ddm_mic2_multinoise_weibull_no_bias_config,
# )


# def get_model_config():
#     """Accessor for model configurations.

#     Returns
#     -------
#     dict
#         Dictionary containing all model configurations.
#     """
#     # TODO: Refactor to load these lazily
#     return {
#         "ddm": get_ddm_config(),
#         "ddm_st": get_ddm_st_config(),
#         "ddm_truncnormt": get_ddm_truncnormt_config(),
#         "ddm_rayleight": get_ddm_rayleight_config(),
#         "ddm_sdv": get_ddm_sdv_config(),
#         "ddm_par2": get_ddm_par2_config(),
#         "ddm_par2_no_bias": get_ddm_par2_no_bias_config(),
#         "ddm_par2_conflict_gamma_no_bias": get_ddm_par2_conflict_gamma_no_bias_config(),
#         "ddm_par2_angle_no_bias": get_ddm_par2_angle_no_bias_config(),
#         "ddm_par2_weibull_no_bias": get_ddm_par2_weibull_no_bias_config(),
#         "ddm_seq2": get_ddm_seq2_config(),
#         "ddm_seq2_no_bias": get_ddm_seq2_no_bias_config(),
#         "ddm_seq2_conflict_gamma_no_bias": get_ddm_seq2_conflict_gamma_no_bias_config(),
#         "ddm_seq2_angle_no_bias": get_ddm_seq2_angle_no_bias_config(),
#         "ddm_seq2_weibull_no_bias": get_ddm_seq2_weibull_no_bias_config(),
#         "ddm_mic2_adj": get_ddm_mic2_adj_config(),
#         "ddm_mic2_adj_no_bias": get_ddm_mic2_adj_no_bias_config(),
#         "ddm_mic2_adj_conflict_gamma_no_bias": get_ddm_mic2_adj_conflict_gamma_no_bias_config(),
#         "ddm_mic2_adj_angle_no_bias": get_ddm_mic2_adj_angle_no_bias_config(),
#         "ddm_mic2_adj_weibull_no_bias": get_ddm_mic2_adj_weibull_no_bias_config(),
#         "ddm_mic2_ornstein": get_ddm_mic2_ornstein_config(),
#         "ddm_mic2_ornstein_no_bias": get_ddm_mic2_ornstein_no_bias_config(),
#         "ddm_mic2_ornstein_conflict_gamma_no_bias": get_ddm_mic2_ornstein_conflict_gamma_no_bias_config(),
#         "ddm_mic2_ornstein_angle_no_bias": get_ddm_mic2_ornstein_angle_no_bias_config(),
#         "ddm_mic2_ornstein_weibull_no_bias": get_ddm_mic2_ornstein_weibull_no_bias_config(),
#         "ddm_mic2_leak": get_ddm_mic2_leak_config(),
#         "ddm_mic2_leak_no_bias": get_ddm_mic2_leak_no_bias_config(),
#         "ddm_mic2_leak_conflict_gamma_no_bias": get_ddm_mic2_leak_conflict_gamma_no_bias_config(),
#         "ddm_mic2_leak_angle_no_bias": get_ddm_mic2_leak_angle_no_bias_config(),
#         "ddm_mic2_leak_weibull_no_bias": get_ddm_mic2_leak_weibull_no_bias_config(),
#         "ddm_mic2_multinoise_no_bias": get_ddm_mic2_multinoise_no_bias_config(),
#         "ddm_mic2_multinoise_conflict_gamma_no_bias": get_ddm_mic2_multinoise_conflict_gamma_no_bias_config(),
#         "ddm_mic2_multinoise_angle_no_bias": get_ddm_mic2_multinoise_angle_no_bias_config(),
#         "ddm_mic2_multinoise_weibull_no_bias": get_ddm_mic2_multinoise_weibull_no_bias_config(),
#     }


__all__ = [
    "get_ddm_config",
#     "get_model_config",
#     "get_angle_config",
#     "get_weibull_config",
#     "get_full_ddm_config",
#     "get_ddm_st_config",
#     "get_ddm_truncnormt_config",
#     "get_ddm_rayleight_config",
#     "get_ddm_sdv_config",
#     "get_ddm_par2_config",
#     "get_ddm_par2_no_bias_config",
#     "get_ddm_par2_conflict_gamma_no_bias_config",
#     "get_ddm_par2_angle_no_bias_config",
#     "get_ddm_par2_weibull_no_bias_config",
#     "get_ddm_seq2_config",
#     "get_ddm_seq2_no_bias_config",
#     "get_ddm_seq2_conflict_gamma_no_bias_config",
#     "get_ddm_seq2_angle_no_bias_config",
#     "get_ddm_seq2_weibull_no_bias_config",
#     "get_ddm_mic2_adj_config",
#     "get_ddm_mic2_adj_no_bias_config",
#     "get_ddm_mic2_adj_conflict_gamma_no_bias_config",
#     "get_ddm_mic2_adj_angle_no_bias_config",
#     "get_ddm_mic2_adj_weibull_no_bias_config",
#     "get_ddm_mic2_ornstein_config",
#     "get_ddm_mic2_ornstein_no_bias_config",
#     "get_ddm_mic2_ornstein_conflict_gamma_no_bias_config",
#     "get_ddm_mic2_ornstein_angle_no_bias_config",
#     "get_ddm_mic2_ornstein_weibull_no_bias_config",
#     "get_ddm_mic2_leak_config",
#     "get_ddm_mic2_leak_no_bias_config",
#     "get_ddm_mic2_leak_conflict_gamma_no_bias_config",
#     "get_ddm_mic2_leak_angle_no_bias_config",
#     "get_ddm_mic2_leak_weibull_no_bias_config",
#     "get_ddm_mic2_multinoise_no_bias_config",
#     "get_ddm_mic2_multinoise_conflict_gamma_no_bias_config",
#     "get_ddm_mic2_multinoise_angle_no_bias_config",
#     "get_ddm_mic2_multinoise_weibull_no_bias_config",
]
