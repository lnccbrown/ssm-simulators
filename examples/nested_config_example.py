"""Example: Using the new nested generator_config structure.

This example demonstrates how to use the new nested config structure
for clearer organization and better maintainability.
"""

from ssms.dataset_generators.lan_mlp import TrainingDataGenerator
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
    get_nested_generator_config,
)
from ssms.config._modelconfig.base import get_default_model_config


def example_1_nested_config_from_helper():
    """Example 1: Get nested config using helper function."""
    print("=" * 70)
    print("Example 1: Using get_nested_generator_config()")
    print("=" * 70)

    # Get nested config (recommended for new code)
    nested_config = get_nested_generator_config("lan")

    print("\nConfig structure:")
    for section in ["pipeline", "estimator", "training", "simulator", "output"]:
        if section in nested_config:
            print(f"\n{section}:")
            for key, value in nested_config[section].items():
                print(f"  {key}: {value}")

    # Use with TrainingDataGenerator
    model_config = get_default_model_config("ddm")
    _gen = TrainingDataGenerator(nested_config, model_config)

    print("\n✅ TrainingDataGenerator created successfully with nested config!")
    print("   (No deprecation warning)")


def example_2_manual_nested_config():
    """Example 2: Create nested config manually."""
    print("\n" + "=" * 70)
    print("Example 2: Creating nested config manually")
    print("=" * 70)

    # Create your own nested config
    custom_nested_config = {
        "pipeline": {
            "n_parameter_sets": 500,  # How many parameter combinations
            "n_subruns": 5,  # Parallel batches
            "n_cpus": 4,  # CPU cores to use
        },
        "estimator": {
            "type": "kde",  # Estimator type
            "bandwidth": 0.1,  # KDE bandwidth (optional)
            "displace_t": False,  # Displace non-decision time
        },
        "training": {
            "mixture_probabilities": [0.8, 0.1, 0.1],  # Mixture weights
            "n_samples_per_param": 2000,  # Samples per param set
            "separate_response_channels": False,  # Channel separation
        },
        "simulator": {
            "delta_t": 0.001,  # Time step
            "max_t": 20.0,  # Max simulation time
            "n_samples": 100_000,  # Samples for analytical estimation
            "smooth_unif": 0.0,  # Smoothing parameter
        },
        "output": {
            "folder": "data/my_training/",  # Output directory
            "pickle_protocol": 4,  # Pickle protocol version
        },
        # Top-level configs (not yet in sections)
        "model": "ddm",
        "bin_pointwise": False,
    }

    print("\nCustom nested config created:")
    print(
        f"  Pipeline: {custom_nested_config['pipeline']['n_parameter_sets']} param sets"
    )
    print(f"  Estimator: {custom_nested_config['estimator']['type']}")
    print(
        f"  Training: {custom_nested_config['training']['n_samples_per_param']} samples/param"
    )

    # Use with TrainingDataGenerator
    model_config = get_default_model_config("ddm")
    _gen = TrainingDataGenerator(custom_nested_config, model_config)

    print("\n✅ TrainingDataGenerator created successfully with custom nested config!")


def example_3_legacy_flat_config():
    """Example 3: Legacy flat config (still works, but deprecated)."""
    print("\n" + "=" * 70)
    print("Example 3: Legacy flat config (triggers deprecation warning)")
    print("=" * 70)

    # Get flat config (legacy, but still works)
    flat_config = get_default_generator_config("lan", nested=False)

    print("\nFlat config (partial):")
    for key in ["n_parameter_sets", "estimator_type", "delta_t", "output_folder"]:
        if key in flat_config:
            print(f"  {key}: {flat_config[key]}")

    # Use with TrainingDataGenerator (will trigger deprecation warning)
    model_config = get_default_model_config("ddm")

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _gen = TrainingDataGenerator(flat_config, model_config)

        if w:
            print("\n⚠️  Deprecation warning triggered:")
            print(f"   {w[0].message}")

    print("\n   Flat config still works, but please migrate to nested structure!")


def example_4_convert_flat_to_nested():
    """Example 4: Convert existing flat config to nested."""
    print("\n" + "=" * 70)
    print("Example 4: Converting flat config to nested")
    print("=" * 70)

    # Start with flat config
    flat_config = {
        "n_parameter_sets": 100,
        "n_cpus": 2,
        "estimator_type": "kde",
        "kde_bandwidth": 0.15,
        "delta_t": 0.001,
        "max_t": 15.0,
        "output_folder": "data/converted/",
        "pickleprotocol": 4,
    }

    print("\nOriginal flat config:")
    for key, value in flat_config.items():
        print(f"  {key}: {value}")

    # Convert to nested
    from ssms.config.config_utils import convert_flat_to_nested

    nested_config = convert_flat_to_nested(flat_config)

    print("\nConverted to nested config:")
    for section in ["pipeline", "estimator", "simulator", "output"]:
        if section in nested_config and nested_config[section]:
            print(f"\n  {section}:")
            for key, value in nested_config[section].items():
                print(f"    {key}: {value}")

    print("\n✅ Conversion complete!")


def example_5_both_structures_work():
    """Example 5: Demonstrate backward compatibility."""
    print("\n" + "=" * 70)
    print("Example 5: Both structures work (backward compatibility)")
    print("=" * 70)

    # Get both versions
    flat_config = get_default_generator_config("lan", nested=False)
    nested_config = get_default_generator_config("lan", nested=True)

    # Create TrainingDataGenerators with both
    model_config = get_default_model_config("ddm")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings for this demo
        _gen_flat = TrainingDataGenerator(flat_config, model_config)
        _gen_nested = TrainingDataGenerator(nested_config, model_config)

    print("\n✅ Both flat and nested configs work!")
    print("   Flat: Uses legacy structure (with deprecation warning)")
    print("   Nested: Uses new structure (no warning, recommended)")


def example_6_benefits_of_nested():
    """Example 6: Why nested structure is better."""
    print("\n" + "=" * 70)
    print("Example 6: Benefits of Nested Structure")
    print("=" * 70)

    print("""
Benefits of the nested structure:

1. **Clear Separation of Concerns**
   - Pipeline settings (n_parameter_sets, n_cpus)
   - Estimator settings (type, bandwidth)
   - Training settings (mixture_probabilities, n_samples_per_param)
   - Simulator settings (delta_t, max_t)
   - Output settings (folder, pickle_protocol)

2. **Easier to Understand**
   - Config sections match component responsibilities
   - Less confusion about which setting affects what

3. **Better for Future Extensions**
   - Easy to add new settings to appropriate sections
   - Cleaner API for custom components

4. **Reduces Redundancy**
   - Components extract only their relevant section
   - No need to pass entire flat config everywhere

5. **Backward Compatible**
   - Existing flat configs still work
   - Gradual migration path
   - Deprecation warnings guide users

Example of improved clarity:

FLAT (confusing):
    config = {
        "n_parameter_sets": 100,
        "estimator_type": "kde",
        "kde_bandwidth": 0.1,
        "kde_displace_t": False,
        "data_mixture_probabilities": [0.8, 0.1, 0.1],
        "n_training_samples_by_parameter_set": 1000,
        "delta_t": 0.001,
        "max_t": 20.0,
        "output_folder": "data/",
        "pickleprotocol": 4,
        # ... which component uses which setting? 🤔
    }

NESTED (clear):
    config = {
        "pipeline": {"n_parameter_sets": 100},
        "estimator": {"type": "kde", "bandwidth": 0.1, "displace_t": False},
        "training": {"mixture_probabilities": [0.8, 0.1, 0.1], "n_samples_per_param": 1000},
        "simulator": {"delta_t": 0.001, "max_t": 20.0},
        "output": {"folder": "data/", "pickle_protocol": 4},
        # Clear organization! ✨
    }
    """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Nested Config Structure Examples")
    print("=" * 70)

    # Run all examples
    example_1_nested_config_from_helper()
    example_2_manual_nested_config()
    example_3_legacy_flat_config()
    example_4_convert_flat_to_nested()
    example_5_both_structures_work()
    example_6_benefits_of_nested()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nRecommendation: Use nested structure for all new code!")
    print("                Use get_nested_generator_config() helper function.")
