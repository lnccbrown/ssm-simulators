# Nested-Only Config Migration Status

## Decision: Remove Backward Compatibility ✅ APPROVED

User requested to **remove backward compatibility** for flat configs and make nested configs the **only** supported format going forward. This forces clarity about config structure from the start.

---

## Progress So Far

### ✅ Phase 1: Entry Point Validation (COMPLETE)

**Changes Made:**
1. `get_default_generator_config()` - Now **always** returns nested structure
2. `DataGenerator.__init__()` - **Validates** that input configs are nested
3. **Clear error message** with migration instructions if flat config detected

**Result:** Users **cannot** pass flat configs anymore - they get a helpful error with migration guide.

---

### 🔄 Phase 2: Internal Component Updates (IN PROGRESS - ~60% complete)

**Components Updated:**
✅ `KDEEstimatorBuilder` - Uses `get_nested_config()`
✅ `pipeline_factory.py` - Uses `get_nested_config()`
✅ `builder_factory.py` - Uses `get_nested_config()`
✅ `pyddm_builder.py` - Uses `get_nested_config()`
✅ `DataGenerator._get_ncpus()` - Uses nested paths
✅ `DataGenerator._save_training_data()` - Uses nested paths
✅ `DataGenerator._generate_mlp_data_via_strategy()` - Uses nested paths
✅ `SimulationPipeline` - Partially updated (simulator params done, filters done)
✅ `MixtureTrainingStrategy` - Partially updated (training params done)

**Components Still Need Work:**
⚠️ `mixture_training_strategy.py` - Some accesses still failing
⚠️ Test fixtures - Need complete nested config setup
⚠️ Other strategies/builders - May have flat accesses

**Conversion Utility:**
✅ `convert_flat_to_nested()` - Handles most keys
⚠️ May need additional mappings (e.g., `data_mixture_probabilities` without `kde_` prefix)

---

## Current Blocker

**Test Failure:** `ValueError: No valid training data generated. All parameter sets were rejected.`

**Root Cause:** Some config keys are not being found in the nested structure, likely because:
1. The converter doesn't handle all variations (e.g., `data_mixture_probabilities` vs `kde_data_mixture_probabilities`)
2. Some components still expect flat paths
3. Test fixtures may not be setting up complete nested configs

---

## Two Paths Forward

### Option A: Continue Manual Migration (Estimated ~2-4 more hours)

**Approach:**
- Systematically update every config access in every component
- Fix test fixtures to use complete nested configs
- Add missing keys to converter
- Verify each component individually

**Pros:**
- Most thorough
- Ensures every component understands nested structure
- No hidden flat accesses remain

**Cons:**
- Time-consuming
- Error-prone (easy to miss accesses)
- Many scattered changes across codebase

---

### Option B: Smart Hybrid Approach (Estimated ~30-60 minutes)

**Approach:**
1. Keep entry point validation (already done ✅)
2. **Don't** require every internal access to use `get_nested_config()`
3. Instead, ensure `get_default_generator_config()` returns a **dual-format** config:
   ```python
   {
       # Nested structure (primary)
       "pipeline": {"n_parameter_sets": 100, "n_cpus": 4},
       "estimator": {"type": "kde"},
       "training": {"n_samples_per_param": 1000},
       "simulator": {"delta_t": 0.001, "max_t": 20.0},
       "output": {"folder": "data/"},

       # Flattened COPIES for internal backward compat (deprecated paths)
       "n_parameter_sets": 100,  # Copy from pipeline
       "n_cpus": 4,              # Copy from pipeline
       "estimator_type": "kde",  # Copy from estimator.type
       # ... etc
   }
   ```

4. Users **must** provide nested configs (validated at entry)
5. Internally, components can use either path during transition
6. Over time, migrate internal accesses to nested (non-breaking)

**Pros:**
- **Immediate** - unblocks tests/usage now
- Users forced to think in nested terms
- Internal migration can happen gradually
- No risk of breaking existing internal code
- Clear deprecation path

**Cons:**
- Temporary duplication in config dict
- Not as "pure" as full migration
- Need to document which paths are public vs internal

---

## Recommendation

**Use Option B** (Smart Hybrid) because:

1. **User goal achieved** - Users must provide nested configs ✅
2. **Unblocks immediately** - Tests pass, code works ✅
3. **Lower risk** - Doesn't require touching every component ✅
4. **Gradual migration** - Internal code can migrate over time ✅
5. **Pragmatic** - Focuses on user-facing API first ✅

The key insight: **The user-facing API is what matters most.** If users are forced to think in nested terms, that's 90% of the benefit. Internal implementation can follow.

---

## Implementation Plan for Option B

### Step 1: Update `convert_flat_to_nested()` to return dual-format

```python
def convert_flat_to_nested(flat_config: dict) -> dict:
    """Convert flat config to nested, WITH flat copies for internal compat."""
    nested = {
        "pipeline": {},
        "estimator": {},
        "training": {},
        "simulator": {},
        "output": {},
    }

    # ... populate nested sections ...

    # ADD: Flattened copies for internal backward compat
    nested.update(_create_flat_copies(nested))

    return nested

def _create_flat_copies(nested: dict) -> dict:
    """Create flattened copies of nested config for internal use."""
    flat = {}

    # Pipeline
    if "pipeline" in nested:
        flat.update(nested["pipeline"])

    # Estimator
    if "estimator" in nested:
        flat["estimator_type"] = nested["estimator"].get("type", "kde")
        if "displace_t" in nested["estimator"]:
            flat["kde_displace_t"] = nested["estimator"]["displace_t"]
        # ... etc

    # Training
    if "training" in nested:
        if "mixture_probabilities" in nested["training"]:
            flat["kde_data_mixture_probabilities"] = nested["training"]["mixture_probabilities"]
            flat["data_mixture_probabilities"] = nested["training"]["mixture_probabilities"]  # Both!
        # ... etc

    # Simulator
    if "simulator" in nested:
        flat.update({k: v for k, v in nested["simulator"].items() if k != "filters"})
        if "filters" in nested["simulator"]:
            flat["simulation_filters"] = nested["simulator"]["filters"]

    # Output
    if "output" in nested:
        flat["output_folder"] = nested["output"].get("folder")
        flat["pickleprotocol"] = nested["output"].get("pickle_protocol")

    return flat
```

### Step 2: Document public vs internal paths

Add to config docstring:
```
Public API (users MUST use these):
- config["pipeline"]["n_parameter_sets"]
- config["estimator"]["type"]
- etc.

Internal paths (DEPRECATED, for backward compat only):
- config["n_parameter_sets"]  # DO NOT USE in new code
- config["estimator_type"]    # DO NOT USE in new code
```

### Step 3: Test and verify

Run all tests - should pass immediately.

### Step 4: Gradual migration (future work)

Over time, update internal components to use nested paths. Not urgent.

---

## Current Files Modified

1. `ssms/config/generator_config/data_generator_config.py` - Always returns nested
2. `ssms/dataset_generators/lan_mlp.py` - Validates nested, uses some nested paths
3. `ssms/config/config_utils.py` - Converter (needs dual-format update)
4. `ssms/dataset_generators/estimator_builders/kde_builder.py` - Uses `get_nested_config()`
5. `ssms/dataset_generators/pipelines/pipeline_factory.py` - Uses `get_nested_config()`
6. `ssms/dataset_generators/pipelines/simulation_pipeline.py` - Uses nested paths
7. `ssms/dataset_generators/strategies/mixture_training_strategy.py` - Partially updated
8. `tests/dataset_generators/test_integration.py` - Updated fixture

---

## Next Step

**Awaiting user decision:**
- Continue with Option A (manual migration of all components)?
- Switch to Option B (smart hybrid with dual-format)?
- Alternative approach?
