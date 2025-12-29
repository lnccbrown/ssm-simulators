# PyDDM Mapper Architecture Clarification

## TL;DR

The PyDDM mapper uses a **registry-based architecture**, NOT a "legacy vs new format" system. All 100+ predefined model configs use the same pattern: specify a name, and the mapper looks up metadata from a central registry.

---

## The Confusion

Previous documentation incorrectly described the mapper as supporting "two formats":
- ❌ "Structured format" (boundary_config/drift_config dicts)
- ❌ "Legacy format" (boundary_name/drift_name fields)

This was **misleading**. There is only one format used by all standard models.

---

## The Reality

### Standard Architecture (Used by All Predefined Models)

**Model configs specify:**
```python
{
    "boundary_name": "angle",
    "boundary": bf.angle,
    "drift_name": "gamma_drift",
    "drift_fun": df.gamma_drift,
    ...
}
```

**Mapper looks up metadata from central registry:**
```python
# In base.boundary_config
"angle": {
    "fun": bf.angle,
    "params": ["theta"],
    "multiplicative": False
}

# In base.drift_config
"gamma_drift": {
    "fun": df.gamma_drift,
    "params": ["shape", "scale", "c"]
}
```

This is **registry lookup**, not "two formats".

### Optional Direct Embedding (Rare, for Custom Runtime Configs)

For programmatically-created models, you CAN bypass the registry:

```python
# Direct embedding (optional)
model_config["boundary_config"] = {
    "fun": my_custom_function,
    "params": ["custom_param"],
    "multiplicative": True
}
```

But this is:
- **Not used** by any predefined model
- **Optional** for custom models
- **Not a "new format"** - just a convenience for runtime model creation

---

## Why This Architecture?

### Benefits of Registry Pattern

1. **DRY (Don't Repeat Yourself)**
   - Metadata defined once in `base.py`
   - Not duplicated across 100+ model configs

2. **Single Source of Truth**
   - All "angle" boundaries behave identically
   - Changes propagate automatically to all models

3. **Maintainability**
   - Fix a bug in one place
   - Improves all models using that boundary/drift

4. **Consistency**
   - No risk of conflicting definitions
   - Guaranteed uniform behavior

5. **Extensibility**
   - Add new boundary/drift type to registry
   - Immediately available to all models

### What We Avoid

**Without registry (hypothetical bad design):**
```python
# Every model would need to duplicate this:
"angle": {
    "boundary_name": "angle",
    "boundary": bf.angle,
    "boundary_params": ["theta"],        # Duplicated!
    "boundary_multiplicative": False,    # Duplicated!
}

"angle_gamma": {
    "boundary_name": "angle",
    "boundary": bf.angle,
    "boundary_params": ["theta"],        # Duplicated AGAIN!
    "boundary_multiplicative": False,    # Duplicated AGAIN!
}

# ... 50 more models with angle boundaries ...
```

This would be:
- ❌ Error-prone (typos, inconsistencies)
- ❌ Hard to maintain (need to update 50 places for one fix)
- ❌ Violates DRY principle
- ❌ Bloated config files

---

## Updated Terminology

### Before (Confusing)
- ✗ "Structured format" vs "Legacy format"
- ✗ "New system" vs "Old system"
- ✗ "Parameter transforms format" vs "Individual fields"

### After (Clear)
- ✓ "Registry lookup" (standard path)
- ✓ "Direct embedding" (optional, for custom configs)
- ✓ "Central metadata registry" (base.boundary_config, base.drift_config)

---

## Code Flow

### Standard Model (e.g., "angle")

```
1. Model config: {"boundary_name": "angle", "boundary": bf.angle}
                  ↓
2. Mapper: "Let me look up 'angle' in base.boundary_config"
                  ↓
3. Registry: {"params": ["theta"], "multiplicative": False}
                  ↓
4. Mapper: "Got it! I'll create a PyDDM boundary using these params"
                  ↓
5. PyDDM: boundary(t, **theta) function ready to use
```

### Custom Runtime Model (rare)

```
1. User code: {"boundary_config": {"fun": my_fn, "params": [...], ...}}
                  ↓
2. Mapper: "Found direct config, skip registry lookup"
                  ↓
3. PyDDM: boundary(t, **theta) function ready to use
```

---

## Implementation in PyDDM Mapper

### `create_boundary_function()` Logic

```python
def create_boundary_function(model_config):
    # Check for direct embedding (rare, custom configs)
    boundary_cfg = model_config.get("boundary_config", None)

    # Standard path: look up in registry
    if boundary_cfg is None:
        boundary_name = model_config.get("boundary_name", "constant")
        boundary_fn = model_config.get("boundary", None)

        if boundary_name != "constant" and boundary_fn is not None:
            # REGISTRY LOOKUP (standard path for all predefined models)
            from ssms.config._modelconfig.base import boundary_config

            if boundary_name in boundary_config:
                boundary_params = boundary_config[boundary_name]["params"]
                is_multiplicative = boundary_config[boundary_name]["multiplicative"]

                boundary_cfg = {
                    "fun": boundary_fn,
                    "params": boundary_params,
                    "multiplicative": is_multiplicative
                }

    # Use boundary_cfg to create PyDDM function
    # ...
```

### `create_drift_function()` Logic

Similar pattern:
1. Check for direct `drift_config` (rare)
2. Fall back to `drift_name` registry lookup (standard)
3. Use metadata to create PyDDM-compatible drift function

---

## Summary

The PyDDM mapper uses a **registry-based architecture** where:

- ✅ All 100+ predefined models use registry lookup
- ✅ Metadata lives in one place (`base.boundary_config`, `base.drift_config`)
- ✅ Direct embedding is an optional convenience, not a "new format"
- ✅ Architecture follows DRY, maintainability, and consistency principles

**There are not "two formats" - there is one architecture with an optional direct-embedding path for edge cases.**
