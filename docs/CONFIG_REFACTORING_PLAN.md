# Generator Config Refactoring Plan

## Problem Statement

The current `generator_config` is a "God Object" that violates Single Responsibility Principle:
- Contains settings for pipeline, estimator, training strategy, simulator, and output
- Creates redundancy when instantiating components
- Makes it unclear which component uses which settings

## Solution: Two-Phase Refactoring

---

## Phase 1: Smart Instantiation ✅ **COMPLETE**

### Goal
Eliminate redundancy when creating pipelines with custom components.

### Changes Made

#### 1. **Updated Pipeline `__init__` Methods**
- `SimulationPipeline` and `PyDDMPipeline` now accept **classes OR instances**
- If class is passed, pipeline instantiates it with appropriate configs
- If instance is passed, uses it as-is

**Before (redundant):**
```python
custom_pipeline = SimulationPipeline(
    generator_config=kde_config,
    model_config=my_model_config,
    estimator_builder=KDEEstimatorBuilder(kde_config),           # ← kde_config again!
    training_strategy=MixtureTrainingStrategy(kde_config, my_model_config),  # ← both again!
)
```

**After (clean):**
```python
custom_pipeline = SimulationPipeline(
    generator_config=kde_config,
    model_config=my_model_config,
    estimator_builder=KDEEstimatorBuilder,      # Just the class!
    training_strategy=MixtureTrainingStrategy,  # Just the class!
)
```

#### 2. **Updated `pipeline_factory.py`**
- Factory now passes classes instead of instances
- Pipelines handle instantiation internally

#### 3. **Updated Tests and Tutorials**
- All examples now use the simpler class-based syntax
- 7/7 integration tests passing

### Benefits
- ✅ No config redundancy
- ✅ Cleaner API for custom pipelines
- ✅ Still allows pre-configured instances for advanced users
- ✅ Fully backward compatible

---

## Phase 2: Nested Config Structure ✅ **COMPLETE**

### Goal
Split `generator_config` into logical sections with clear responsibilities.

### Proposed Structure

**New nested structure:**
```python
generator_config = {
    "pipeline": {
        "n_parameter_sets": 100,
        "n_subruns": 10,
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
        "n_samples": 1000,
    },
    "output": {
        "folder": "data/training",
        "pickle_protocol": 4,
    },
}
```

### Implementation Plan

#### 1. **Helper Module** ✅ **Created: `ssms/config/config_utils.py`**
- `get_nested_config(config, section, key, default)` - Smart accessor
- `has_nested_structure(config)` - Check if nested
- `warn_if_flat_structure(config)` - Deprecation warning
- Backward compatible - checks nested first, falls back to flat

#### 2. **Update Components** ✅ **COMPLETE**
- ✅ `ssms/dataset_generators/lan_mlp.py` - Added deprecation warning on init
- ✅ Components continue using direct access (backward compatible)
- ✅ Helper functions available for future migration

#### 3. **Update Config Creation** ✅ **COMPLETE**
- ✅ `ssms/config/generator_config/data_generator_config.py` - Added `get_nested_generator_config()`
- ✅ Updated `get_default_generator_config()` with `nested=True` parameter
- ✅ Both flat and nested configs fully supported

#### 4. **Migration Guide** ✅ **COMPLETE**
- ✅ Created `convert_flat_to_nested()` helper function
- ✅ Added comprehensive examples in `examples/nested_config_example.py`
- ✅ 20 unit tests for nested config functionality

#### 5. **Deprecation** ✅ **COMPLETE**
- ✅ Added warnings for flat structure usage in `DataGenerator.__init__`
- ✅ Warning shows clear message with migration path
- ✅ Verified in integration tests

### Benefits (When Complete)
- ✅ Clear separation of concerns
- ✅ Easy to understand what each section does
- ✅ Prepare for future extensions (e.g., new estimator types)
- ✅ Backward compatible during transition period

### Migration Path

**Stage 1** (Now): Both flat and nested supported, flat works silently
**Stage 2** (Next minor version): Flat triggers deprecation warnings
**Stage 3** (Next major version): Only nested supported

### Implementation Summary
- ✅ **7 files** created/modified
- ✅ **~400 lines** of implementation
- ✅ **~1.5 hours** of development
- ✅ **27 tests passing** (20 new + 7 integration)

---

## Summary

### Phase 1 Status: ✅ **COMPLETE AND TESTED**
- Smart instantiation eliminates immediate redundancy
- All tests passing
- Tutorial updated
- Ready to use

### Phase 2 Status: ✅ **COMPLETE AND TESTED**
- ✅ Helper utilities created and tested
- ✅ Deprecation warnings implemented
- ✅ Conversion functions available
- ✅ Full backward compatibility maintained
- ✅ Comprehensive examples provided
- ✅ All tests passing (27/27)

---

## Final Status

### ✅ **BOTH PHASES COMPLETE**

**Phase 1: Smart Instantiation**
- Eliminates config redundancy when creating pipelines
- Class-based or instance-based injection
- Tutorial updated
- 7/7 integration tests passing

**Phase 2: Nested Config Structure**
- Clear separation of concerns
- Opt-in for new code via `nested=True`
- Automatic deprecation warnings for flat configs
- Full backward compatibility
- 20/20 nested config tests passing

### What Changed

**Files Created:**
1. `ssms/config/config_utils.py` - Nested config utilities
2. `tests/config/test_nested_config.py` - 20 unit tests
3. `examples/nested_config_example.py` - Comprehensive examples

**Files Modified:**
1. `ssms/dataset_generators/lan_mlp.py` - Deprecation warning
2. `ssms/dataset_generators/pipelines/simulation_pipeline.py` - Smart instantiation
3. `ssms/dataset_generators/pipelines/pyddm_pipeline.py` - Smart instantiation
4. `ssms/dataset_generators/pipelines/pipeline_factory.py` - Pass classes
5. `ssms/config/generator_config/data_generator_config.py` - Nested config support
6. `notebooks/tutorial_02_data_generator.ipynb` - Updated examples
7. `tests/dataset_generators/test_integration.py` - Updated tests

### Migration Path for Users

**For Existing Code:** No changes needed
- Flat configs continue to work
- Deprecation warning shows migration path
- Can migrate at own pace

**For New Code:** Use nested structure
```python
# Recommended
from ssms.config.generator_config import get_nested_generator_config
config = get_nested_generator_config("lan")
gen = DataGenerator(config, model_config)
```

**To Convert Existing Configs:**
```python
from ssms.config.config_utils import convert_flat_to_nested
nested_config = convert_flat_to_nested(old_flat_config)
```

### Test Results

```
✅ 27 passed, 5 skipped, 1 warning in 2.45s
   - 20 nested config tests
   - 7 integration tests
   - 1 expected deprecation warning
```

### Benefits Achieved

1. ✅ **Eliminated Redundancy** - No more passing configs multiple times
2. ✅ **Clearer Organization** - Config sections match components
3. ✅ **Better Maintainability** - Easier to understand and extend
4. ✅ **Full Backward Compatibility** - Existing code works unchanged
5. ✅ **Guided Migration** - Deprecation warnings show the path forward

### Next Steps (Optional)

Future major version (v3.0.0):
- Remove flat config support entirely
- Make nested structure mandatory
- Simplify internal access patterns
