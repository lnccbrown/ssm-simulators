"""
This module defines a data generator class for use with LANs.
The class defined below can be used to generate training data
compatible with the expectations of LANs.
"""

import logging
import uuid
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from ssms.dataset_generators.protocols import (
    DataGenerationPipelineProtocol,
)

import pickle
import numpy as np
import psutil
from pathos.multiprocessing import ProcessingPool as Pool

from ssms.config import KDE_NO_DISPLACE_T

logger = logging.getLogger(__name__)


class TrainingDataGenerator:  # noqa: N801
    """The TrainingDataGenerator() class is used to generate training data
      for various likelihood approximators.

    Attributes
    ----------
        generator_config: dict
            Configuration dictionary for the data generator.
            (For an example load ssms.config.get_lan_config())
        model_config: dict
            Configuration dictionary for the model to be simulated.
            (For an example load ssms.config.model_config['ddm'])
        _generation_pipeline: DataGenerationPipelineProtocol
            Strategy for orchestrating the complete data generation workflow.

    Methods
    -------
        generate_data_training_uniform(save=False, verbose=True)
            Generates training data for LANs.
        _get_ncpus()
            Helper function for determining the number of
            cpus to use for parallelization.

    Returns
    -------
        TrainingDataGenerator object

    Notes
    -----
    The class supports dependency injection of generation_pipeline, enabling
    complete customization of the data generation workflow. By default, it
    auto-creates the appropriate strategy based on generator_config settings
    (KDE-based simulation or PyDDM analytical methods).
    """

    def __init__(
        self,
        config: Union[dict, DataGenerationPipelineProtocol, None] = None,
        model_config: Optional[dict] = None,
    ):
        """Initialize data generator class.

        Arguments
        ---------
        config: dict or DataGenerationPipelineProtocol
            Either:
            - A dictionary with generator configuration (e.g., from
              ssms.config.get_default_generator_config()). The TrainingDataGenerator
              will auto-create the appropriate strategy.
            - A DataGenerationPipelineProtocol instance for complete custom
              control over the data generation workflow.
        model_config: dict, optional
            Configuration dictionary for the model to be simulated.
            (For an example load ssms.config.model_config['ddm'])
            This serves as the single source of truth for model specifications,
            parameter bounds, transforms, and custom drift/boundary functions.

            **Required** when config is a dict (generator_config).
            **Optional** when config is a pipeline (extracted from pipeline if not provided).

        Raises
        ------
        ValueError
            If config is None, or if model_config is None when config is a dict,
            or if model_config cannot be extracted from a pipeline.

        Returns
        -------
        TrainingDataGenerator object

        Notes
        -----
        **Simple Usage**:
        Pass a generator_config dict as the first argument. The TrainingDataGenerator
        will auto-create the appropriate strategy based on 'estimator_type'
        in the config ('kde' for simulation-based, 'pyddm' for analytical).

        **Advanced Usage**:
        Pass a custom DataGenerationPipelineProtocol instance to completely
        control the workflow (parameter sampling, simulation, estimation, etc.).
        In this case, model_config can be omitted (will use pipeline's config).

        The pipeline internally manages:
        - Parameter sampling (uniform, Sobol, custom)
        - Simulation or analytical PDF computation
        - Likelihood estimation (KDE, PyDDM analytical)
        - Training data structuring (ResampleMixture, etc.)

        For custom drift/boundary functions, use ModelConfigBuilder to create
        a custom model_config before passing it to TrainingDataGenerator.

        Examples
        --------
        Simple (dict config):
            >>> gen = TrainingDataGenerator(generator_config, model_config)
            >>> data = gen.generate_data_training()

        Advanced (custom pipeline, model_config from pipeline):
            >>> from ssms.dataset_generators.pipelines import SimulationPipeline
            >>> custom_pipeline = SimulationPipeline(
            ...     generator_config, model_config,
            ...     estimator_builder=MyCustomBuilder(...),
            ...     training_strategy=MyCustomStrategy(...),
            ... )
            >>> gen = TrainingDataGenerator(custom_pipeline)  # model_config optional!
            >>> data = gen.generate_data_training()
        """
        # INIT -----------------------------------------
        # Check if config is a dict (generator_config) or a pipeline object
        if config is None:
            raise ValueError("No config specified")

        # Determine if config is a dict or a pipeline
        if isinstance(config, dict):
            # Config is a dictionary - treat as generator_config
            # In this case, model_config is REQUIRED
            if model_config is None:
                raise ValueError(
                    "model_config is required when config is a dictionary. "
                    "Pass both generator_config and model_config."
                )

            # Validate that config uses nested structure
            from ssms.config.config_utils import has_nested_structure

            if not has_nested_structure(config):
                raise ValueError(
                    "Flat generator_config structure is no longer supported. "
                    "Please use the nested structure with 'pipeline', 'estimator', "
                    "'training', 'simulator', and 'output' sections.\n\n"
                    "To migrate, use:\n"
                    "  from ssms.config.config_utils import convert_flat_to_nested\n"
                    "  nested_config = convert_flat_to_nested(old_flat_config)\n\n"
                    "Or get a nested config directly:\n"
                    "  from ssms.config.generator_config import get_default_generator_config\n"
                    "  config = get_default_generator_config('lan')  # Always nested now"
                )

            self.generator_config = deepcopy(config)
            self.model_config = deepcopy(model_config)
            generation_pipeline = None  # Will auto-create later
        else:
            # Config is a pipeline object
            generation_pipeline = config

            # model_config is OPTIONAL when passing a pipeline
            # Extract from pipeline if not provided
            if model_config is None:
                # Use pipeline's model_config
                if not hasattr(generation_pipeline, "model_config"):
                    raise ValueError(
                        "Pipeline object must have 'model_config' attribute, "
                        "or model_config must be provided as argument."
                    )
                self.model_config = deepcopy(generation_pipeline.model_config)
            else:
                # Both provided - warn and use pipeline's
                if (
                    hasattr(generation_pipeline, "model_config")
                    and generation_pipeline.model_config is not None
                ):
                    logger.warning(
                        "model_config argument provided along with a custom pipeline. "
                        "The pipeline's model_config will be used, and the argument will be ignored."
                    )
                    self.model_config = deepcopy(generation_pipeline.model_config)
                else:
                    # Pipeline doesn't have model_config, use the provided one
                    self.model_config = deepcopy(model_config)

            # Extract generator_config from pipeline
            self.generator_config = getattr(
                generation_pipeline, "generator_config", None
            )

        # Handle config-specific setup only if we have generator_config dict
        if isinstance(config, dict):
            # Account for deadline if in model name
            # Note: This mutates the model_config - ideally this should be handled
            # at config creation time, but kept for backward compatibility
            if "deadline" in self.generator_config["model"]:
                self.model_config["params"].append("deadline")
                if isinstance(self.model_config["param_bounds"], list):
                    self.model_config["param_bounds"][0].append(0.001)
                    self.model_config["param_bounds"][1].append(10)
                    self.model_config["default_params"].append(10)
                    self.model_config["name"] += "_deadline"
                    self.model_config["n_params"] += 1
                    # Update param_bounds_dict to include deadline
                    self.model_config["param_bounds_dict"]["deadline"] = (0.001, 10)
                elif isinstance(self.model_config["param_bounds"], dict):
                    self.model_config["param_bounds"]["deadline"] = (0.001, 10)
                    self.model_config["default_params"].append(10)
                    self.model_config["name"] += "_deadline"
                    self.model_config["n_params"] += 1
                    # Update param_bounds_dict to match
                    self.model_config["param_bounds_dict"]["deadline"] = (0.001, 10)

            # Ensure estimator section exists and set default displace_t
            if "estimator" not in self.generator_config:
                self.generator_config["estimator"] = {}
            if "displace_t" not in self.generator_config["estimator"]:
                self.generator_config["estimator"]["displace_t"] = False

            if (
                self.generator_config["estimator"]["displace_t"]
                and self.model_config["name"].split("_deadline")[0] in KDE_NO_DISPLACE_T
            ):
                warnings.warn(
                    f"displace_t is True, but model is in {KDE_NO_DISPLACE_T}."
                    " Overriding setting to False",
                    stacklevel=2,
                )
                self.generator_config["estimator"]["displace_t"] = False

            # self._build_simulator()

        # Always call _get_ncpus if we have generator_config (from dict or strategy)
        if self.generator_config is not None:
            self._get_ncpus()

        # Set up generation pipeline
        if generation_pipeline is not None:
            # Strategy was passed directly
            self._generation_pipeline = generation_pipeline
        else:
            # Auto-create strategy from configs
            from ssms.dataset_generators.pipelines.pipeline_factory import (
                create_data_generation_pipeline,
            )

            # Factory creates strategy with all required components internally
            # (estimator builder, training strategy, parameter sampler, etc.)
            self._generation_pipeline = create_data_generation_pipeline(
                generator_config=self.generator_config,
                model_config=self.model_config,
            )

        # Make output folder if not already present (only if we have generator_config)
        if self.generator_config is not None:
            output_folder = Path(self.generator_config["output"]["folder"])
            output_folder.mkdir(parents=True, exist_ok=True)

    def _get_ncpus(self):
        """Get the number cpus to use for parallelization."""
        from ssms.config.config_utils import get_nested_config

        # Get number of cpus from nested config
        n_cpus_config = get_nested_config(
            self.generator_config, "pipeline", "n_cpus", default="all"
        )

        if n_cpus_config == "all":
            n_cpus = psutil.cpu_count(logical=False)
        else:
            n_cpus = n_cpus_config

        # Update nested config
        if "pipeline" not in self.generator_config:
            self.generator_config["pipeline"] = {}
        self.generator_config["pipeline"]["n_cpus"] = n_cpus

    def _save_training_data(self, data: dict) -> None:
        """Save training data to disk as pickle file.

        Args:
            data: Dictionary containing training data to save

        Notes:
            Creates output folder if it doesn't exist. Filename includes
            a unique UUID to prevent overwriting.
        """
        output_folder = Path(self.generator_config["output"]["folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        full_file_name = output_folder / f"training_data_{uuid.uuid1().hex}.pickle"
        logger.info("Writing to file: %s", full_file_name)

        with full_file_name.open("wb") as file:
            pickle.dump(
                data,
                file,
                protocol=self.generator_config["output"]["pickle_protocol"],
            )
        logger.info("Data saved successfully")

    def generate_data_training(self, save: bool = False, verbose: bool = False):
        """Generates training data for LANs.

        Arguments
        ---------
            save: bool
                If True, the generated data is saved to disk.
            verbose: bool
                If True, progress is printed to the console.

        Returns
        -------
            data: dict
                Dictionary containing the generated data.

        Notes
        -----
            Phase 4 refactoring: This method now delegates to the
            generation_pipeline, which handles the complete workflow from
            parameter sampling to training data generation. This eliminates
            wasteful simulations for analytical methods (PyDDM) and provides
            a clean, modular architecture.
        """
        # Phase 4: Always use generation pipeline
        return self._generate_mlp_data_via_strategy(save, verbose)

    def _generate_mlp_data_via_strategy(
        self, save: bool = False, verbose: bool = False
    ) -> dict:
        """Generate MLP training data using injected generation pipeline.

        This method implements Phase 4 refactoring: it delegates to the
        generation_pipeline for orchestrating the complete workflow. The strategy
        handles simulation (if needed), filtering, likelihood estimation, and
        training data generation.

        Args:
            save: Whether to save generated data to disk
            verbose: Whether to log progress

        Returns:
            Dictionary containing generated training data

        Notes:
            This is the Phase 4 implementation that provides:
            - Clean separation between simulation-based and analytical workflows
            - No wasteful simulations for PyDDM
            - Common parallelization wrapper for all strategies
        """
        # Generate seeds for each parameter set
        seeds_2 = np.random.choice(
            400000000, size=self.generator_config["pipeline"]["n_parameter_sets"]
        )

        # Inits
        subrun_n = (
            self.generator_config["pipeline"]["n_parameter_sets"]
            // self.generator_config["pipeline"]["n_subruns"]
        )

        # Common parallelization wrapper (works for all strategies)
        out_list = []
        for i in range(self.generator_config["pipeline"]["n_subruns"]):
            if verbose:
                logger.debug(
                    "generation round: %d of %d",
                    i + 1,
                    self.generator_config["pipeline"]["n_subruns"],
                )

            start_idx = i * subrun_n
            end_idx = (i + 1) * subrun_n

            if self.generator_config["pipeline"]["n_cpus"] > 1:
                with Pool(
                    processes=self.generator_config["pipeline"]["n_cpus"] - 1
                ) as pool:
                    # Map strategy.generate_for_parameter_set over parameter indices
                    results = pool.map(
                        self._generation_pipeline.generate_for_parameter_set,
                        list(range(start_idx, end_idx)),
                        list(seeds_2[start_idx:end_idx]),
                    )
                    out_list += results
            else:
                if verbose:
                    logger.info("No Multiprocessing, since only one cpu requested!")
                for parameter_sampling_seed, seed in zip(
                    range(start_idx, end_idx), seeds_2[start_idx:end_idx]
                ):
                    result = self._generation_pipeline.generate_for_parameter_set(
                        parameter_sampling_seed, seed
                    )
                    out_list.append(result)

        # Filter successful results
        successful_results = [r for r in out_list if r["success"]]

        if len(successful_results) == 0:
            raise ValueError(
                "No valid training data generated. All parameter sets were rejected."
            )

        if verbose and len(successful_results) < len(out_list):
            logger.warning(
                "Rejected %d/%d parameter sets",
                len(out_list) - len(successful_results),
                len(out_list),
            )

        # Extract data from successful results
        # Note: Some fields may be None for PyDDM strategy
        # (e.g., cpn_labels, binned data)
        data = {}

        # Helper function to safely concatenate arrays (skip None values)
        def safe_concatenate(key):
            arrays = [
                r["data"][key] for r in successful_results if r["data"][key] is not None
            ]
            if len(arrays) == 0:
                return None
            return np.concatenate(arrays).astype(np.float32)

        # Extract all unique keys from successful results
        # This makes the code future-proof for new pipeline outputs
        all_keys = set()
        for result in successful_results:
            all_keys.update(result["data"].keys())

        # Concatenate all data arrays
        for key in all_keys:
            data[key] = safe_concatenate(key)

        # Add metadata
        data.update(
            {
                "generator_config": self.generator_config,
                "model_config": self.model_config,
            }
        )

        if save:
            self._save_training_data(data)
        return data
