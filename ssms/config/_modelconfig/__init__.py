"""Model configuration module for SSM simulators."""

from .ddm import (
    get_ddm_config,
    get_ddm_legacy_config,
)
from .ds_conflict_drift import (
    get_ds_conflict_drift_angle_config,
    get_ds_conflict_drift_config,
    get_ds_conflict_stimflexons_drift_angle_config,
    get_ds_conflict_stimflexons_drift_config,
)
from .full_ddm import (
    get_full_ddm_config,
    get_full_ddm_rv_config,
)
from .lca import (
    get_lca_3_config,
    get_lca_4_config,
    get_lca_no_bias_3_config,
    get_lca_no_bias_4_config,
    get_lca_no_bias_angle_3_config,
    get_lca_no_bias_angle_4_config,
    get_lca_no_z_3_config,
    get_lca_no_z_4_config,
    get_lca_no_z_angle_3_config,
    get_lca_no_z_angle_4_config,
)
from .levy import get_levy_angle_config, get_levy_config
from .tradeoff import (
    get_tradeoff_angle_no_bias_config,
    get_tradeoff_conflict_gamma_no_bias_config,
    get_tradeoff_no_bias_config,
    get_tradeoff_weibull_no_bias_config,
)

from .angle import get_angle_config
from .weibull import get_weibull_config
from .ddm_par2 import (
    get_ddm_par2_angle_no_bias_config,
    get_ddm_par2_config,
    get_ddm_par2_conflict_gamma_no_bias_config,
    get_ddm_par2_no_bias_config,
    get_ddm_par2_weibull_no_bias_config,
)
from .ddm_random import (
    get_ddm_rayleight_config,
    get_ddm_sdv_config,
    get_ddm_st_config,
    get_ddm_truncnormt_config,
)
from .ddm_seq2 import (
    get_ddm_seq2_angle_no_bias_config,
    get_ddm_seq2_config,
    get_ddm_seq2_conflict_gamma_no_bias_config,
    get_ddm_seq2_no_bias_config,
    get_ddm_seq2_weibull_no_bias_config,
)
from .dev_rlwm_lba import (
    get_dev_rlwm_lba_pw_v1_config,
    get_dev_rlwm_lba_race_v1_config,
    get_dev_rlwm_lba_race_v2_config,
)
from .gamma_drift import (
    get_gamma_drift_angle_config,
    get_gamma_drift_config,
)
from .lba import (
    get_lba2_config,
    get_lba3_config,
    get_lba_3_vs_constraint_config,
    get_lba_angle_3_config,
    get_lba_angle_3_vs_constraint_config,
)
from .mic2 import (
    get_ddm_mic2_adj_angle_no_bias_config,
    get_ddm_mic2_adj_config,
    get_ddm_mic2_adj_conflict_gamma_no_bias_config,
    get_ddm_mic2_adj_no_bias_config,
    get_ddm_mic2_adj_weibull_no_bias_config,
    get_ddm_mic2_leak_angle_no_bias_config,
    get_ddm_mic2_leak_config,
    get_ddm_mic2_leak_conflict_gamma_no_bias_config,
    get_ddm_mic2_leak_no_bias_config,
    get_ddm_mic2_leak_weibull_no_bias_config,
    get_ddm_mic2_ornstein_angle_no_bias_config,
    get_ddm_mic2_ornstein_config,
    get_ddm_mic2_ornstein_conflict_gamma_no_bias_config,
    get_ddm_mic2_ornstein_no_bias_config,
    get_ddm_mic2_ornstein_weibull_no_bias_config,
)
from .mic2.multinoise import (
    get_ddm_mic2_multinoise_angle_no_bias_config,
    get_ddm_mic2_multinoise_conflict_gamma_no_bias_config,
    get_ddm_mic2_multinoise_no_bias_config,
    get_ddm_mic2_multinoise_weibull_no_bias_config,
)
from .ornstein import (
    get_ornstein_angle_config,
    get_ornstein_config,
)
from .race import (
    get_race_2_config,
    get_race_3_config,
    get_race_4_config,
    get_race_no_bias_2_config,
    get_race_no_bias_3_config,
    get_race_no_bias_4_config,
    get_race_no_bias_angle_2_config,
    get_race_no_bias_angle_3_config,
    get_race_no_bias_angle_4_config,
    get_race_no_z_2_config,
    get_race_no_z_3_config,
    get_race_no_z_4_config,
    get_race_no_z_angle_2_config,
    get_race_no_z_angle_3_config,
    get_race_no_z_angle_4_config,
)
from .shrink import (
    get_shrink_spot_config,
    get_shrink_spot_extended_config,
    get_shrink_spot_simple_config,
    get_shrink_spot_simple_extended_config,
)

from .exgauss import (
    get_exgauss_config,
    get_exgauss_race_config
)

from .shifted_wald import (
    get_shifted_wald_config,
    get_shifted_wald_race_config
)

def get_model_config():
    """Accessor for model configurations.

    Returns
    -------
    dict
        Dictionary containing all model configurations.
    """
    # TODO: Refactor to load these lazily
    return {
        "ddm": get_ddm_config(),
        "ddm_st": get_ddm_st_config(),
        "ddm_truncnormt": get_ddm_truncnormt_config(),
        "ddm_rayleight": get_ddm_rayleight_config(),
        "ddm_sdv": get_ddm_sdv_config(),
        "ddm_par2": get_ddm_par2_config(),
        "ddm_par2_no_bias": get_ddm_par2_no_bias_config(),
        "ddm_par2_conflict_gamma_no_bias": get_ddm_par2_conflict_gamma_no_bias_config(),
        "ddm_par2_angle_no_bias": get_ddm_par2_angle_no_bias_config(),
        "ddm_par2_weibull_no_bias": get_ddm_par2_weibull_no_bias_config(),
        "ddm_seq2": get_ddm_seq2_config(),
        "ddm_seq2_no_bias": get_ddm_seq2_no_bias_config(),
        "ddm_seq2_conflict_gamma_no_bias": get_ddm_seq2_conflict_gamma_no_bias_config(),
        "ddm_seq2_angle_no_bias": get_ddm_seq2_angle_no_bias_config(),
        "ddm_seq2_weibull_no_bias": get_ddm_seq2_weibull_no_bias_config(),
        "ddm_mic2_adj": get_ddm_mic2_adj_config(),
        "ddm_mic2_adj_no_bias": get_ddm_mic2_adj_no_bias_config(),
        "ddm_mic2_adj_conflict_gamma_no_bias": get_ddm_mic2_adj_conflict_gamma_no_bias_config(),
        "ddm_mic2_adj_angle_no_bias": get_ddm_mic2_adj_angle_no_bias_config(),
        "ddm_mic2_adj_weibull_no_bias": get_ddm_mic2_adj_weibull_no_bias_config(),
        "ddm_mic2_ornstein": get_ddm_mic2_ornstein_config(),
        "ddm_mic2_ornstein_no_bias": get_ddm_mic2_ornstein_no_bias_config(),
        "ddm_mic2_ornstein_no_bias_no_lowdim_noise": get_ddm_mic2_ornstein_no_bias_config(),
        "ddm_mic2_ornstein_conflict_gamma_no_bias": get_ddm_mic2_ornstein_conflict_gamma_no_bias_config(),
        "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise": get_ddm_mic2_ornstein_conflict_gamma_no_bias_config(),
        "ddm_mic2_ornstein_angle_no_bias": get_ddm_mic2_ornstein_angle_no_bias_config(),
        "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise": get_ddm_mic2_ornstein_angle_no_bias_config(),
        "ddm_mic2_ornstein_weibull_no_bias": get_ddm_mic2_ornstein_weibull_no_bias_config(),
        "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise": get_ddm_mic2_ornstein_weibull_no_bias_config(),
        "ddm_mic2_leak": get_ddm_mic2_leak_config(),
        "ddm_mic2_leak_no_bias": get_ddm_mic2_leak_no_bias_config(),
        "ddm_mic2_leak_no_bias_no_lowdim_noise": get_ddm_mic2_leak_no_bias_config(),
        "ddm_mic2_leak_conflict_gamma_no_bias": get_ddm_mic2_leak_conflict_gamma_no_bias_config(),
        "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise": get_ddm_mic2_leak_conflict_gamma_no_bias_config(),
        "ddm_mic2_leak_angle_no_bias": get_ddm_mic2_leak_angle_no_bias_config(),
        "ddm_mic2_leak_angle_no_bias_no_lowdim_noise": get_ddm_mic2_leak_angle_no_bias_config(),
        "ddm_mic2_leak_weibull_no_bias": get_ddm_mic2_leak_weibull_no_bias_config(),
        "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise": get_ddm_mic2_leak_weibull_no_bias_config(),
        "ddm_mic2_multinoise_no_bias": get_ddm_mic2_multinoise_no_bias_config(),
        "ddm_mic2_multinoise_conflict_gamma_no_bias": get_ddm_mic2_multinoise_conflict_gamma_no_bias_config(),
        "ddm_mic2_multinoise_angle_no_bias": get_ddm_mic2_multinoise_angle_no_bias_config(),
        "ddm_mic2_multinoise_weibull_no_bias": get_ddm_mic2_multinoise_weibull_no_bias_config(),
        "full_ddm": get_full_ddm_config(),
        "full_ddm_rv": get_full_ddm_rv_config(),
        "levy": get_levy_config(),
        "levy_angle": get_levy_angle_config(),
        "angle": get_angle_config(),
        "weibull": get_weibull_config(),
        "gamma_drift": get_gamma_drift_config(),
        "shrink_spot": get_shrink_spot_config(),
        "shrink_spot_extended": get_shrink_spot_extended_config(),
        "shrink_spot_simple": get_shrink_spot_simple_config(),
        "shrink_spot_simple_extended": get_shrink_spot_simple_extended_config(),
        "gamma_drift_angle": get_gamma_drift_angle_config(),
        "ds_conflict_drift": get_ds_conflict_drift_config(),
        "ds_conflict_drift_angle": get_ds_conflict_drift_angle_config(),
        "ds_conflict_stimflexons_drift": get_ds_conflict_stimflexons_drift_config(),
        "ds_conflict_stimflexons_drift_angle": get_ds_conflict_stimflexons_drift_angle_config(),
        "ornstein": get_ornstein_config(),
        "ornstein_angle": get_ornstein_angle_config(),
        "race_2": get_race_2_config(),
        "race_no_bias_2": get_race_no_bias_2_config(),
        "race_no_z_2": get_race_no_z_2_config(),
        "race_no_bias_angle_2": get_race_no_bias_angle_2_config(),
        "race_no_z_angle_2": get_race_no_z_angle_2_config(),
        "race_3": get_race_3_config(),
        "race_no_bias_3": get_race_no_bias_3_config(),
        "race_no_z_3": get_race_no_z_3_config(),
        "race_no_bias_angle_3": get_race_no_bias_angle_3_config(),
        "race_no_z_angle_3": get_race_no_z_angle_3_config(),
        "race_4": get_race_4_config(),
        "race_no_bias_4": get_race_no_bias_4_config(),
        "race_no_z_4": get_race_no_z_4_config(),
        "race_no_bias_angle_4": get_race_no_bias_angle_4_config(),
        "race_no_z_angle_4": get_race_no_z_angle_4_config(),
        "dev_rlwm_lba_pw_v1": get_dev_rlwm_lba_pw_v1_config(),
        "dev_rlwm_lba_race_v1": get_dev_rlwm_lba_race_v1_config(),
        "dev_rlwm_lba_race_v2": get_dev_rlwm_lba_race_v2_config(),
        "lba2": get_lba2_config(),
        "lba3": get_lba3_config(),
        "lba_3_vs_constraint": get_lba_3_vs_constraint_config(),
        "lba_angle_3_vs_constraint": get_lba_angle_3_vs_constraint_config(),
        "lba_angle_3": get_lba_angle_3_config(),
        "lca_3": get_lca_3_config(),
        "lca_no_bias_3": get_lca_no_bias_3_config(),
        "lca_no_z_3": get_lca_no_z_3_config(),
        "lca_no_bias_angle_3": get_lca_no_bias_angle_3_config(),
        "lca_no_z_angle_3": get_lca_no_z_angle_3_config(),
        "lca_4": get_lca_4_config(),
        "lca_no_bias_4": get_lca_no_bias_4_config(),
        "lca_no_z_4": get_lca_no_z_4_config(),
        "lca_no_bias_angle_4": get_lca_no_bias_angle_4_config(),
        "lca_no_z_angle_4": get_lca_no_z_angle_4_config(),
        "tradeoff_no_bias": get_tradeoff_no_bias_config(),
        "tradeoff_angle_no_bias": get_tradeoff_angle_no_bias_config(),
        "tradeoff_weibull_no_bias": get_tradeoff_weibull_no_bias_config(),
        "tradeoff_conflict_gamma_no_bias": get_tradeoff_conflict_gamma_no_bias_config(),
        "weibull_cdf": get_weibull_config(),
        "full_ddm2": get_full_ddm_config(),
        "ddm_legacy": get_ddm_legacy_config(),
        "exgauss": get_exgauss_config(),
        "exgauss_race": get_exgauss_race_config(),
        "shifted_wald": get_shifted_wald_config(),
        "shifted_wald_race": get_shifted_wald_race_config(),
    }


__all__ = [
    "get_model_config",
    "get_ddm_config",
    "get_angle_config",
    "get_weibull_config",
    "get_full_ddm_config",
    "get_ddm_st_config",
    "get_ddm_truncnormt_config",
    "get_ddm_rayleight_config",
    "get_ddm_sdv_config",
    "get_ddm_par2_config",
    "get_ddm_par2_no_bias_config",
    "get_ddm_par2_conflict_gamma_no_bias_config",
    "get_ddm_par2_angle_no_bias_config",
    "get_ddm_par2_weibull_no_bias_config",
    "get_ddm_seq2_config",
    "get_ddm_seq2_no_bias_config",
    "get_ddm_seq2_conflict_gamma_no_bias_config",
    "get_ddm_seq2_angle_no_bias_config",
    "get_ddm_seq2_weibull_no_bias_config",
    "get_ddm_mic2_adj_config",
    "get_ddm_mic2_adj_no_bias_config",
    "get_ddm_mic2_adj_conflict_gamma_no_bias_config",
    "get_ddm_mic2_adj_angle_no_bias_config",
    "get_ddm_mic2_adj_weibull_no_bias_config",
    "get_ddm_mic2_ornstein_config",
    "get_ddm_mic2_ornstein_no_bias_config",
    "get_ddm_mic2_ornstein_conflict_gamma_no_bias_config",
    "get_ddm_mic2_ornstein_angle_no_bias_config",
    "get_ddm_mic2_ornstein_weibull_no_bias_config",
    "get_ddm_mic2_leak_config",
    "get_ddm_mic2_leak_no_bias_config",
    "get_ddm_mic2_leak_conflict_gamma_no_bias_config",
    "get_ddm_mic2_leak_angle_no_bias_config",
    "get_ddm_mic2_leak_weibull_no_bias_config",
    "get_ddm_mic2_multinoise_no_bias_config",
    "get_ddm_mic2_multinoise_conflict_gamma_no_bias_config",
    "get_ddm_mic2_multinoise_angle_no_bias_config",
    "get_ddm_mic2_multinoise_weibull_no_bias_config",
]
