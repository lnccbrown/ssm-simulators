# Nested Config Migration Guide

## Overview

The generator configuration system has been enhanced with a new **nested structure** that provides clearer organization and better separation of concerns. This guide explains the changes and how to migrate your code.

## TL;DR

**For existing code:** No changes needed - flat configs still work (with deprecation warning)

**For new code:** Use `get_nested_generator_config()` for cleaner, more organized configs

```python
# New way (recommended)
from ssms.config.generator_config import get_nested_generator_config
config = get_nested_generator_config("lan")

# Old way (still works, but deprecated)
from ssms.config.generator_config import get_default_generator_config
config = get_default_generator_config("lan")  # Triggers deprecation warning
```

---

## What Changed?

### The Problem: "God Object" Config

The old flat config mixed concerns from different components:

```python
# Old flat structure - everything mixed together 😕
config = {
    "n_parameter_sets": 100,        # Pipeline setting
    "n_cpus": 4,                    # Pipeline setting
    "estimator_type": "kde",        # Estimator setting
    "kde_bandwidth": 0.1,           # Estimator setting
    "kde_data_mixture_probabilities": [0.8, 0.1, 0.1],  # Training setting
    "n_training_samples_by_parameter_set": 1000,  # Training setting
    "delta_t": 0.001,               # Simulator setting
    "max_t": 20.0,                  # Simulator setting
    "output_folder": "data/",       # Output setting
    "pickleprotocol": 4,            # Output setting
}
```

**Issues:**
- Hard to understand which component uses which setting
- No clear organization
- Difficult to extend
- Setting names had component prefixes (e.g., `kde_data_mixture_probabilities`)

### The Solution: Nested Structure

New nested structure groups settings by component:

```python
# New nested structure - clear organization ✨
config = {
    "pipeline": {
        "n_parameter_sets": 100,
        "n_cpus": 4,
    },
    "estimator": {
        "type": "kde",
        "bandwidth": 0.1,
    },
    "training": {
        "mixture_probabilities": [0.8, 0.1, 0.1],
        "n_samples_per_param": 1000,
    },
    "simulator": {
        "delta_t": 0.001,
        "max_t": 20.0,
    },
    "output": {
        "folder": "data/",
        "pickle_protocol": 4,
    },
}
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Easier to understand and maintain
- ✅ Cleaner setting names (no prefixes needed)
- ✅ Better for future extensions

---

## Migration Strategies

### Strategy 1: Do Nothing (For Now)

**Best for:** Existing code, stable projects

Your existing code will continue to work unchanged:

```python
# This still works
config = get_default_generator_config("lan")
gen = DataGenerator(config, model_config)
```

You'll see a deprecation warning:
```
DeprecationWarning: Flat generator_config structure is deprecated and will be
removed in a future version. Please use the nested structure...
```

**Action:** Plan to migrate when convenient.

---

### Strategy 2: Opt-In to Nested (Recommended)

**Best for:** New projects, active development

Use the new `nested=True` parameter or convenience function:

```python
# Option A: Use convenience function
from ssms.config.generator_config import get_nested_generator_config
config = get_nested_generator_config("lan")

# Option B: Use nested parameter
from ssms.config.generator_config import get_default_generator_config
config = get_default_generator_config("lan", nested=True)

# Use as normal (no warning)
gen = DataGenerator(config, model_config)
```

**Action:** Update new code to use nested configs.

---

### Strategy 3: Convert Existing Configs

**Best for:** Migrating custom configs

Use the conversion utility for existing flat configs:

```python
from ssms.config.config_utils import convert_flat_to_nested

# Your existing flat config
old_config = {
    "n_parameter_sets": 500,
    "estimator_type": "kde",
    "delta_t": 0.001,
    # ... other flat settings
}

# Convert to nested
new_config = convert_flat_to_nested(old_config)

# Now nested!
print(new_config["pipeline"]["n_parameter_sets"])  # 500
print(new_config["estimator"]["type"])  # "kde"
```

**Action:** Run conversion once, save as new config files.

---

### Strategy 4: Manual Creation

**Best for:** Fine-tuned custom configs

Create nested configs from scratch for maximum clarity:

```python
# Create your own nested config
my_config = {
    "pipeline": {
        "n_parameter_sets": 1000,
        "n_subruns": 20,
        "n_cpus": 8,
    },
    "estimator": {
        "type": "kde",
        "bandwidth": 0.15,
    },
    "training": {
        "mixture_probabilities": [0.7, 0.2, 0.1],
        "n_samples_per_param": 2000,
    },
    "simulator": {
        "delta_t": 0.0005,
        "max_t": 25.0,
    },
    "output": {
        "folder": "data/my_experiment/",
        "pickle_protocol": 4,
    },
    # Top-level (not yet nested)
    "model": "ddm",
    "bin_pointwise": False,
}

gen = DataGenerator(my_config, model_config)
```

**Action:** Write new configs in nested format directly.

---

## Key Mappings: Flat → Nested

### Pipeline Settings
| Flat Key | Nested Location |
|----------|----------------|
| `n_parameter_sets` | `pipeline.n_parameter_sets` |
| `n_subruns` | `pipeline.n_subruns` |
| `n_cpus` | `pipeline.n_cpus` |

### Estimator Settings
| Flat Key | Nested Location |
|----------|----------------|
| `estimator_type` | `estimator.type` |
| `kde_bandwidth` | `estimator.bandwidth` |
| `kde_displace_t` | `estimator.displace_t` |
| `use_pyddm_pdf` | `estimator.use_pyddm_pdf` |

### Training Settings
| Flat Key | Nested Location |
|----------|----------------|
| `kde_data_mixture_probabilities` | `training.mixture_probabilities` |
| `n_training_samples_by_parameter_set` | `training.n_samples_per_param` |
| `separate_response_channels` | `training.separate_response_channels` |

### Simulator Settings
| Flat Key | Nested Location |
|----------|----------------|
| `delta_t` | `simulator.delta_t` |
| `max_t` | `simulator.max_t` |
| `n_samples` | `simulator.n_samples` |
| `smooth_unif` | `simulator.smooth_unif` |
| `simulation_filters` | `simulator.filters` |

### Output Settings
| Flat Key | Nested Location |
|----------|----------------|
| `output_folder` | `output.folder` |
| `pickleprotocol` | `output.pickle_protocol` |

### Top-Level (Not Yet Nested)
| Key | Location |
|-----|----------|
| `model` | `model` (top-level) |
| `bin_pointwise` | `bin_pointwise` (top-level) |

---

## Backward Compatibility

### What Still Works

✅ **All existing code** - No breaking changes
✅ **Flat configs** - Continue to function
✅ **All tutorials** - Work with both structures
✅ **All tests** - Pass with both structures

### What's Different

⚠️ **Deprecation Warning** - Shown when using flat configs
ℹ️ **Recommended Approach** - Use nested for new code
📅 **Future Removal** - Flat structure will be removed in v3.0.0

---

## Examples

### Example 1: Quick Start with Nested Config

```python
from ssms.dataset_generators.lan_mlp import DataGenerator
from ssms.config.generator_config import get_nested_generator_config
from ssms.config._modelconfig.base import get_default_model_config

# Get nested config (no deprecation warning)
config = get_nested_generator_config("lan")
model_config = get_default_model_config("ddm")

# Create generator
gen = DataGenerator(config, model_config)

# Generate data
training_data = gen.generate_data_training(save=True)
```

### Example 2: Custom Nested Config

```python
# Define custom nested config
custom_config = {
    "pipeline": {
        "n_parameter_sets": 500,
        "n_subruns": 10,
        "n_cpus": 4,
    },
    "estimator": {
        "type": "kde",
        "bandwidth": 0.12,
    },
    "training": {
        "mixture_probabilities": [0.9, 0.05, 0.05],
        "n_samples_per_param": 1500,
    },
    "simulator": {
        "delta_t": 0.001,
        "max_t": 20.0,
        "n_samples": 100_000,
    },
    "output": {
        "folder": "data/custom_experiment/",
        "pickle_protocol": 4,
    },
    "model": "angle",
    "bin_pointwise": False,
}

# Use directly
gen = DataGenerator(custom_config, model_config)
```

### Example 3: Converting Existing Config File

```python
import json
from ssms.config.config_utils import convert_flat_to_nested

# Load existing flat config
with open("old_config.json", "r") as f:
    old_config = json.load(f)

# Convert to nested
new_config = convert_flat_to_nested(old_config)

# Save nested config
with open("new_config.json", "w") as f:
    json.dump(new_config, f, indent=2)

print("Config converted and saved!")
```

---

## Timeline

### Current (v2.x)
- ✅ Both flat and nested supported
- ⚠️ Deprecation warnings for flat configs
- 📚 Migration guide available

### Future (v3.0)
- ❌ Flat structure removed
- ✅ Only nested structure supported
- 🎯 Cleaner internal implementation

**Recommendation:** Migrate to nested structure during the v2.x lifecycle to ensure smooth transition to v3.0.

---

## FAQ

### Q: Do I need to change my code immediately?
**A:** No. Existing code will continue to work, but you'll see deprecation warnings.

### Q: What if I have many config files?
**A:** Use `convert_flat_to_nested()` to batch-convert them, or convert gradually as you update code.

### Q: Can I mix flat and nested configs?
**A:** The system handles both, but we recommend choosing one style per project for consistency.

### Q: Will nested configs make my code more verbose?
**A:** Slightly, but the improved clarity and organization outweigh the minor increase in lines.

### Q: How do I access nested config values in my custom components?
**A:** Use the `get_nested_config()` helper:
```python
from ssms.config.config_utils import get_nested_config

# Works with both flat and nested
n_params = get_nested_config(config, "pipeline", "n_parameter_sets", default=100)
```

### Q: When will flat configs stop working?
**A:** Planned for removal in v3.0.0 (no specific date yet). You'll have plenty of time to migrate.

---

## Getting Help

- **Examples:** See `examples/nested_config_example.py`
- **Tests:** See `tests/config/test_nested_config.py`
- **Documentation:** See `docs/CONFIG_REFACTORING_PLAN.md`

---

## Summary

✅ **Nested structure** is the future of generator configs
✅ **Backward compatible** - existing code works
✅ **Easy migration** - helpers and examples provided
✅ **Better design** - clearer, more maintainable

**Start using nested configs today for cleaner, more maintainable code!**
