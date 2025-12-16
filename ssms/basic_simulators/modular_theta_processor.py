"""
Modular theta processor using transformation pipelines.

This module provides the ModularThetaProcessor class which applies registered
theta transformations in sequence based on model configuration.
"""

from typing import Any

import numpy as np

from ssms.basic_simulators.theta_processor import AbstractThetaProcessor
from ssms.basic_simulators.theta_transforms import (
    ApplyMapping,
    ColumnStackParameters,
    DeleteParameters,
    ExpandDimension,
    LambdaTransformation,
    RenameParameter,
    SetZeroArray,
    ThetaProcessorRegistry,
)


class ModularThetaProcessor(AbstractThetaProcessor):
    """Modular theta processor using transformation pipelines.

    This processor applies a sequence of theta transformations based on the
    model name. Transformations are registered in a ThetaProcessorRegistry
    which maps model names (and model families) to transformation pipelines.

    The processor can be customized by providing a custom registry or by
    adding additional transformations after initialization.

    Parameters
    ----------
    registry : ThetaProcessorRegistry or None, optional
        Registry containing model → transformations mappings.
        If None, uses the default registry with all built-in models.

    Examples
    --------
    >>> # Use default registry
    >>> processor = ModularThetaProcessor()
    >>> theta = processor.process_theta(theta, model_config, n_trials)
    >>>
    >>> # Use custom registry
    >>> custom_registry = ThetaProcessorRegistry()
    >>> custom_registry.register_model("my_model", [...])
    >>> processor = ModularThetaProcessor(registry=custom_registry)
    """

    def __init__(self, registry: ThetaProcessorRegistry | None = None):
        """Initialize processor with registry."""
        self.registry = (
            registry if registry is not None else self._build_default_registry()
        )

    def process_theta(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """Process theta by applying registered transformations.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of theta parameters
        model_config : dict[str, Any]
            Model configuration dictionary
        n_trials : int
            Number of trials

        Returns
        -------
        dict[str, Any]
            Processed theta parameters
        """
        model_name = model_config.get("name", "")
        transformations = self.registry.get_processor(model_name)

        # Apply transformations in sequence
        for transformation in transformations:
            theta = transformation.apply(theta, model_config, n_trials)

        return theta

    @staticmethod
    def _build_default_registry() -> ThetaProcessorRegistry:
        """Build registry with all default model transformations.

        This method creates a registry populated with transformation pipelines
        for all built-in models. It's called automatically if no registry is
        provided to __init__.

        Returns
        -------
        ThetaProcessorRegistry
            Registry with all default model registrations
        """
        registry = ThetaProcessorRegistry()

        # ================================================================
        # SINGLE-PARTICLE MODELS (No transformations needed)
        # ================================================================
        no_transform_models = [
            "glob",
            "ddm",
            "angle",
            "weibull",
            "ddm_hddm_base",
            "ddm_legacy",
            "levy",
            "levy_angle",
            "full_ddm",
            "full_ddm_legacy",
            "full_ddm_hddm_base",
            "ornstein",
            "ornstein_angle",
            "gamma_drift",
            "gamma_drift_angle",
        ]
        for model in no_transform_models:
            registry.register_model(model, [])

        # ================================================================
        # DYNAMIC DRIFT MODELS
        # ================================================================
        dynamic_drift_models = [
            "conflict_ds",
            "conflict_ds_angle",
            "conflict_dsstimflex",
            "conflict_dsstimflex_angle",
            "conflict_stimflex",
            "conflict_stimflex_angle",
            "conflict_stimflexrel1",
            "conflict_stimflexrel1_angle",
            "conflict_stimflexrel1_leak",
            "conflict_stimflexrel1_leak2",
            "shrink_spot",
            "shrink_spot_simple",
            "shrink_spot_extended",
            "shrink_spot_extended_angle",
            "shrink_spot_simple_extended",
        ]
        for model in dynamic_drift_models:
            registry.register_model(
                model,
                [
                    SetZeroArray("v"),
                ],
            )

        # Dual drift models (2 drift values)
        dual_drift_models = [
            "conflict_stimflex_leak2_drift",
            "conflict_stimflex_leak2_drift_angle",
        ]
        for model in dual_drift_models:
            registry.register_model(
                model,
                [
                    SetZeroArray("v", shape=(None, 2)),
                ],
            )

        # ================================================================
        # RANDOM VARIABLE MODELS
        # ================================================================

        registry.register_model(
            "ddm_st",
            [
                # Fixed parameters from config
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update(
                        {
                            "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                            "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                        }
                    )
                    or theta,
                    name="set_fixed_params",
                ),
                # Apply mapping to st → t_dist
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        )

        registry.register_model(
            "ddm_rayleight",
            [
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update(
                        {
                            "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                            "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                            "t": (
                                np.ones(n) * cfg["simulator_fixed_params"]["t"]
                            ).astype(np.float32),
                        }
                    )
                    or theta,
                    name="set_fixed_params_and_t",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        )

        registry.register_model(
            "ddm_truncnormt",
            [
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update(
                        {
                            "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                            "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                            "t": np.array([0], dtype=np.float32),
                        }
                    )
                    or theta,
                    name="set_fixed_params_and_zero_t",
                ),
                ApplyMapping("mt", "t_dist", "t_dist", additional_sources=["st"]),
            ],
        )

        registry.register_model(
            "ddm_sdv",
            [
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update(
                        {
                            "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                            "t_dist": cfg["simulator_fixed_params"]["t_dist"],
                        }
                    )
                    or theta,
                    name="set_fixed_dists",
                ),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        )

        registry.register_model(
            "full_ddm_rv",
            [
                ApplyMapping("sz", "z_dist", "z_dist"),
                ApplyMapping("st", "t_dist", "t_dist"),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        )

        # ================================================================
        # LBA MODELS
        # ================================================================

        # Helper for setting t to zeros
        set_zero_t = LambdaTransformation(
            lambda theta, cfg, n: theta.update({"t": np.zeros(n).astype(np.float32)})
            or theta,
            name="set_zero_t",
        )

        registry.register_model(
            "lba2",
            [
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update({"nact": 2}) or theta,
                    name="set_nact_2",
                ),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                RenameParameter("A", "z", lambda x: np.expand_dims(x, axis=1)),
                RenameParameter("b", "a", lambda x: np.expand_dims(x, axis=1)),
                DeleteParameters(["A", "b"]),  # Explicitly delete A and b
                set_zero_t,
            ],
        )

        registry.register_model(
            "lba3",
            [
                LambdaTransformation(
                    lambda theta, cfg, n: theta.update({"nact": 3}) or theta,
                    name="set_nact_3",
                ),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                RenameParameter("A", "z", lambda x: np.expand_dims(x, axis=1)),
                RenameParameter("b", "a", lambda x: np.expand_dims(x, axis=1)),
                DeleteParameters(["A", "b"]),  # Explicitly delete A and b
                set_zero_t,
            ],
        )

        # LBA with constraints (non-angle)
        registry.register_model(
            "lba_3_vs_constraint",
            [
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["a", "z"]),
                set_zero_t,
            ],
        )

        # LBA with constraints (angle variants)
        for model in ["lba_angle_3_vs_constraint", "lba_angle_3"]:
            registry.register_model(
                model,
                [
                    ColumnStackParameters(
                        ["v0", "v1", "v2"], "v", delete_sources=False
                    ),
                    ExpandDimension(["a", "z", "theta"]),
                    set_zero_t,
                ],
            )

        # RLWM LBA models
        for model in ["dev_rlwm_lba_race_v1", "dev_rlwm_lba_race_v2"]:
            registry.register_model(
                model,
                [
                    ColumnStackParameters(
                        ["vRL0", "vRL1", "vRL2"], "vRL", delete_sources=False
                    ),
                    ColumnStackParameters(
                        ["vWM0", "vWM1", "vWM2"], "vWM", delete_sources=False
                    ),
                    ExpandDimension(["a", "z"]),
                    set_zero_t,
                ],
            )

        registry.register_model(
            "dev_rlwm_lba_pw_v1",
            [
                ColumnStackParameters(
                    ["vRL0", "vRL1", "vRL2"], "vRL", delete_sources=False
                ),
                ColumnStackParameters(
                    ["vWM0", "vWM1", "vWM2"], "vWM", delete_sources=False
                ),
                ExpandDimension(["a", "z", "tWM"]),
                set_zero_t,
            ],
        )

        # ================================================================
        # RACE MODELS (using family registration)
        # ================================================================

        # 2-choice race models
        registry.register_family(
            "race_2",
            lambda m: m.startswith("race_") and m.endswith("_2") and "angle" not in m,
            [
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        )

        # 2-choice race with angle
        registry.register_family(
            "race_angle_2",
            lambda m: m.startswith("race_") and m.endswith("_2") and "angle" in m,
            [
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        )

        # Specific variants
        registry.register_model(
            "race_2",
            [
                ColumnStackParameters(["z0", "z1"], "z", delete_sources=False),
                ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        )

        # No-bias variants set z to shared value
        for model in ["race_no_bias_2", "race_no_bias_angle_2"]:
            registry.register_model(
                model,
                [
                    LambdaTransformation(
                        lambda theta, cfg, n: theta.update(
                            {"z": np.column_stack([theta["z"], theta["z"]])}
                        )
                        or theta,
                        name="duplicate_z_2",
                    ),
                    ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                    ExpandDimension(["t", "a"]),
                ],
            )

        # No-z variants set z to zeros
        for model in ["race_no_z_2", "race_no_z_angle_2"]:
            registry.register_model(
                model,
                [
                    SetZeroArray("z", shape=(None, 2)),
                    ColumnStackParameters(["v0", "v1"], "v", delete_sources=False),
                    ExpandDimension(["t", "a"]),
                ],
            )

        # ================================================================
        # 3-CHOICE RACE MODELS
        # ================================================================

        registry.register_model(
            "race_3",
            [
                ColumnStackParameters(["z0", "z1", "z2"], "z", delete_sources=False),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a"]),
            ],
        )

        for model in ["race_no_bias_3", "race_no_bias_angle_3"]:
            registry.register_model(
                model,
                [
                    LambdaTransformation(
                        lambda theta, cfg, n: theta.update(
                            {"z": np.column_stack([theta["z"], theta["z"], theta["z"]])}
                        )
                        or theta,
                        name="duplicate_z_3",
                    ),
                    ColumnStackParameters(
                        ["v0", "v1", "v2"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a"]),
                ],
            )

        for model in ["race_no_z_3", "race_no_z_angle_3"]:
            registry.register_model(
                model,
                [
                    SetZeroArray("z", shape=(None, 3)),
                    ColumnStackParameters(
                        ["v0", "v1", "v2"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a"]),
                ],
            )

        # ================================================================
        # 4-CHOICE RACE MODELS
        # ================================================================

        registry.register_model(
            "race_4",
            [
                ColumnStackParameters(
                    ["z0", "z1", "z2", "z3"], "z", delete_sources=False
                ),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a"]),
            ],
        )

        for model in ["race_no_bias_4", "race_no_bias_angle_4"]:
            registry.register_model(
                model,
                [
                    LambdaTransformation(
                        lambda theta, cfg, n: theta.update(
                            {"z": np.column_stack([theta["z"]] * 4)}
                        )
                        or theta,
                        name="duplicate_z_4",
                    ),
                    ColumnStackParameters(
                        ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a"]),
                ],
            )

        for model in ["race_no_z_4", "race_no_z_angle_4"]:
            registry.register_model(
                model,
                [
                    SetZeroArray("z", shape=(None, 4)),
                    ColumnStackParameters(
                        ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a"]),
                ],
            )

        # ================================================================
        # LCA MODELS
        # ================================================================

        # Poisson race (2-choice, Poisson finishing times)
        registry.register_model(
            "poisson_race",
            [
                ColumnStackParameters(["r0", "r1"], "r", delete_sources=False),
                ColumnStackParameters(["k0", "k1"], "k", delete_sources=False),
                ExpandDimension(["t"]),
            ],
        )

        # LCA 3-choice
        registry.register_model(
            "lca_3",
            [
                ColumnStackParameters(["z0", "z1", "z2"], "z", delete_sources=False),
                ColumnStackParameters(["v0", "v1", "v2"], "v", delete_sources=False),
                ExpandDimension(["t", "a", "g", "b"]),
            ],
        )

        for model in ["lca_no_bias_3", "lca_no_bias_angle_3"]:
            registry.register_model(
                model,
                [
                    LambdaTransformation(
                        lambda theta, cfg, n: theta.update(
                            {"z": np.column_stack([theta["z"]] * 3)}
                        )
                        or theta,
                        name="duplicate_z_3",
                    ),
                    ColumnStackParameters(
                        ["v0", "v1", "v2"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a", "g", "b"]),
                ],
            )

        for model in ["lca_no_z_3", "lca_no_z_angle_3"]:
            registry.register_model(
                model,
                [
                    SetZeroArray("z", shape=(None, 3)),
                    ColumnStackParameters(
                        ["v0", "v1", "v2"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a", "g", "b"]),
                ],
            )

        # LCA 4-choice
        registry.register_model(
            "lca_4",
            [
                ColumnStackParameters(
                    ["z0", "z1", "z2", "z3"], "z", delete_sources=False
                ),
                ColumnStackParameters(
                    ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                ),
                ExpandDimension(["t", "a", "g", "b"]),
            ],
        )

        for model in ["lca_no_bias_4", "lca_no_bias_angle_4"]:
            registry.register_model(
                model,
                [
                    LambdaTransformation(
                        lambda theta, cfg, n: theta.update(
                            {"z": np.column_stack([theta["z"]] * 4)}
                        )
                        or theta,
                        name="duplicate_z_4",
                    ),
                    ColumnStackParameters(
                        ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a", "g", "b"]),
                ],
            )

        for model in ["lca_no_z_4", "lca_no_z_angle_4"]:
            registry.register_model(
                model,
                [
                    SetZeroArray("z", shape=(None, 4)),
                    ColumnStackParameters(
                        ["v0", "v1", "v2", "v3"], "v", delete_sources=False
                    ),
                    ExpandDimension(["t", "a", "g", "b"]),
                ],
            )

        # ================================================================
        # SEQUENTIAL/PARALLEL MODELS (4-choice hierarchical)
        # ================================================================

        # Common transformations for no-bias variants
        add_z_defaults = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                        "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                        "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                    }
                )
                or theta
            ),
            name="add_z_defaults",
        )

        # Sequential models (ddm_seq2)
        for model in [
            "ddm_seq2_no_bias",
            "ddm_seq2_angle_no_bias",
            "ddm_seq2_weibull_no_bias",
            "ddm_seq2_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults])

        # Parallel models (ddm_par2)
        for model in [
            "ddm_par2_no_bias",
            "ddm_par2_angle_no_bias",
            "ddm_par2_weibull_no_bias",
            "ddm_par2_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults])

        # MIC2 models - adjusted
        add_mic2_adj_params = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "s_pre_high_level_choice": np.tile(
                            np.array([1.0], dtype=np.float32), n
                        ),
                        "g": np.tile(np.array([0.0], dtype=np.float32), n),
                    }
                )
                or theta
            ),
            name="add_mic2_adj_params",
        )

        registry.register_model("ddm_mic2_adj", [add_mic2_adj_params])

        for model in [
            "ddm_mic2_adj_no_bias",
            "ddm_mic2_adj_angle_no_bias",
            "ddm_mic2_adj_weibull_no_bias",
            "ddm_mic2_adj_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults, add_mic2_adj_params])

        # MIC2 Ornstein variants
        add_ornstein_params = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "s_pre_high_level_choice": np.tile(
                            np.array([1.0], dtype=np.float32), n
                        )
                    }
                )
                or theta
            ),
            name="add_ornstein_params",
        )

        add_ornstein_params_no_lowdim = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "s_pre_high_level_choice": np.tile(
                            np.array([0.0], dtype=np.float32), n
                        )
                    }
                )
                or theta
            ),
            name="add_ornstein_params_no_lowdim",
        )

        registry.register_model("ddm_mic2_ornstein", [add_ornstein_params])

        for model in [
            "ddm_mic2_ornstein_no_bias",
            "ddm_mic2_ornstein_angle_no_bias",
            "ddm_mic2_ornstein_weibull_no_bias",
            "ddm_mic2_ornstein_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults, add_ornstein_params])

        for model in [
            "ddm_mic2_ornstein_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_angle_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_weibull_no_bias_no_lowdim_noise",
            "ddm_mic2_ornstein_conflict_gamma_no_bias_no_lowdim_noise",
        ]:
            registry.register_model(
                model, [add_z_defaults, add_ornstein_params_no_lowdim]
            )

        # MIC2 Leak variants
        add_leak_params = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "g": np.tile(np.array([2.0], dtype=np.float32), n),
                        "s_pre_high_level_choice": np.tile(
                            np.array([1.0], dtype=np.float32), n
                        ),
                    }
                )
                or theta
            ),
            name="add_leak_params",
        )

        add_leak_params_no_lowdim = LambdaTransformation(
            lambda theta, cfg, n: (
                theta.update(
                    {
                        "g": np.tile(np.array([2.0], dtype=np.float32), n),
                        "s_pre_high_level_choice": np.tile(
                            np.array([0.0], dtype=np.float32), n
                        ),
                    }
                )
                or theta
            ),
            name="add_leak_params_no_lowdim",
        )

        registry.register_model("ddm_mic2_leak", [add_leak_params])

        for model in [
            "ddm_mic2_leak_no_bias",
            "ddm_mic2_leak_angle_no_bias",
            "ddm_mic2_leak_weibull_no_bias",
            "ddm_mic2_leak_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults, add_leak_params])

        for model in [
            "ddm_mic2_leak_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_angle_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_weibull_no_bias_no_lowdim_noise",
            "ddm_mic2_leak_conflict_gamma_no_bias_no_lowdim_noise",
        ]:
            registry.register_model(model, [add_z_defaults, add_leak_params_no_lowdim])

        # MIC2 multinoise variants
        for model in [
            "ddm_mic2_multinoise_no_bias",
            "ddm_mic2_multinoise_angle_no_bias",
            "ddm_mic2_multinoise_weibull_no_bias",
            "ddm_mic2_multinoise_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults])

        # Tradeoff models
        for model in [
            "tradeoff_no_bias",
            "tradeoff_angle_no_bias",
            "tradeoff_weibull_no_bias",
            "tradeoff_conflict_gamma_no_bias",
        ]:
            registry.register_model(model, [add_z_defaults])

        return registry
