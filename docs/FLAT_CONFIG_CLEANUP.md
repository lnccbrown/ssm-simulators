# Flat Config Support Cleanup

## Summary

After completing the migration to nested-only configs, removed all unnecessary flat config support code while keeping essential migration utilities.

**Date:** December 29, 2025
**Test Results:** 29/29 passing ✅

---

## What Was Removed

### 1. Removed from `config_utils.py`

#### Removed Functions
- ❌ `warn_if_flat_structure()` - No longer needed (we reject, not warn)
- ❌ `_get_flat_key()` - Helper for flat fallback (no longer needed)

#### Removed Functionality
- ❌ Flat config fallback logic in `get_nested_config()`
  - Previously: Tried nested first, then fell back to flat
  - Now: Only checks nested structure, returns default if not found

#### Removed Imports
- ❌ `import warnings` - No longer issuing warnings

#### Updated Docstrings
- ✅ Module docstring - Removed "backward-compatible" language
- ✅ `get_nested_config()` - Removed flat fallback examples
- ✅ `has_nested_structure()` - Clarified it's for validation, not compatibility

---

## What Was Kept

### Essential Functions in `config_utils.py`

#### ✅ `get_nested_config(config, section, key, default=None)`
**Purpose:** Get values from nested config structure
**Kept because:** Used throughout codebase for clean nested access
**Simplified:** No longer has flat fallback logic

```python
# Works only with nested configs now
config = {"pipeline": {"n_parameter_sets": 100}}
value = get_nested_config(config, "pipeline", "n_parameter_sets")  # 100
```

#### ✅ `has_nested_structure(config)`
**Purpose:** Check if config has nested structure
**Kept because:** Used for validation in `DataGenerator`
**Updated:** Docstring clarifies it's for validation, not compatibility

```python
# Validates config format
nested = {"pipeline": {}}
flat = {"n_parameter_sets": 100}

has_nested_structure(nested)  # True
has_nested_structure(flat)    # False - triggers error in DataGenerator
```

#### ✅ `convert_flat_to_nested(flat_config)`
**Purpose:** Convert legacy flat configs to nested
**Kept because:** Essential migration utility for users
**Unchanged:** Still handles all key mappings

```python
# Migration helper
old_config = {"n_parameter_sets": 100, "estimator_type": "kde"}
new_config = convert_flat_to_nested(old_config)
# Returns: {"pipeline": {"n_parameter_sets": 100}, "estimator": {"type": "kde"}}
```

---

## Test Updates

### Removed Tests
- ❌ `test_warn_if_flat_structure_warns()` - Function removed
- ❌ `test_warn_if_flat_structure_silent_for_nested()` - Function removed
- ❌ `test_get_nested_config_from_flat_structure()` - No flat fallback
- ❌ `test_get_nested_config_nested_takes_precedence()` - No precedence needed

### Added/Updated Tests
- ✅ `test_get_nested_config_returns_none_for_missing_section()` - Clean behavior
- ✅ `test_get_nested_config_only_works_with_nested()` - Verifies no fallback
- ✅ `test_has_nested_structure_detects_flat()` - Validation testing
- ✅ `test_flat_config_rejected_in_practice()` - End-to-end rejection

### Test Results
```
======================== 29 passed, 5 skipped in 2.50s =========================

✅ 22 config tests passing
✅ 7 integration tests passing
```

---

## Code Size Reduction

### `config_utils.py`
**Before:** 208 lines
**After:** 164 lines
**Reduction:** 44 lines (-21%)

### Removed Code Breakdown
- 30 lines: `_get_flat_key()` function with mappings
- 10 lines: `warn_if_flat_structure()` function
- 4 lines: Flat fallback logic in `get_nested_config()`

---

## API Before vs After

### Before Cleanup (Confusing)

```python
from ssms.config.config_utils import get_nested_config, warn_if_flat_structure

# Could work with both flat and nested (confusing!)
flat_config = {"n_parameter_sets": 100}
nested_config = {"pipeline": {"n_parameter_sets": 100}}

# Both worked with fallback logic
get_nested_config(flat_config, "pipeline", "n_parameter_sets")  # 100 (fallback)
get_nested_config(nested_config, "pipeline", "n_parameter_sets")  # 100 (direct)

# Warnings were issued
warn_if_flat_structure(flat_config)  # DeprecationWarning
```

### After Cleanup (Clear)

```python
from ssms.config.config_utils import get_nested_config, has_nested_structure

# Only works with nested (clear!)
nested_config = {"pipeline": {"n_parameter_sets": 100}}
flat_config = {"n_parameter_sets": 100}

# Only nested works
get_nested_config(nested_config, "pipeline", "n_parameter_sets")  # 100 ✅
get_nested_config(flat_config, "pipeline", "n_parameter_sets")     # None ✅

# Validation is explicit
has_nested_structure(nested_config)  # True ✅
has_nested_structure(flat_config)    # False ✅

# Flat configs rejected at entry point
DataGenerator(flat_config, model_config)  # ValueError ✅
```

---

## Benefits of Cleanup

### 1. Clearer Intent ✅
- No ambiguity about what's supported
- No "magic" fallback behavior
- Functions do exactly what they say

### 2. Simpler Code ✅
- 44 fewer lines to maintain
- No complex mapping logic in `get_nested_config()`
- Fewer edge cases to test

### 3. Better Error Messages ✅
- Flat configs fail at entry point (DataGenerator)
- Clear error message with migration instructions
- No confusing "it works but warns" state

### 4. Easier to Understand ✅
- `get_nested_config()` only works with nested configs (obvious!)
- `has_nested_structure()` is purely for validation
- `convert_flat_to_nested()` is clearly a migration tool

### 5. Lower Maintenance Burden ✅
- Less code to maintain
- Fewer tests to keep updated
- Clearer separation of concerns

---

## Migration Path Still Clear

Users who need to migrate flat configs have clear tools:

### Step 1: Identify flat configs
```python
from ssms.config.config_utils import has_nested_structure

if not has_nested_structure(my_config):
    print("Config needs migration!")
```

### Step 2: Convert
```python
from ssms.config.config_utils import convert_flat_to_nested

new_config = convert_flat_to_nested(old_config)
```

### Step 3: Use
```python
from ssms.dataset_generators.lan_mlp import DataGenerator

gen = DataGenerator(new_config, model_config)  # Works!
```

---

## Files Modified

1. ✅ `ssms/config/config_utils.py` - Removed flat support, kept essentials
2. ✅ `tests/config/test_nested_config.py` - Removed obsolete tests, added validation tests

---

## Summary

**Removed:**
- Flat config fallback logic
- Deprecation warnings
- Helper functions for flat configs
- 44 lines of unnecessary code

**Kept:**
- Nested config accessor (`get_nested_config`)
- Validation helper (`has_nested_structure`)
- Migration utility (`convert_flat_to_nested`)

**Result:**
- ✅ Clearer, simpler API
- ✅ All 29 tests passing
- ✅ Migration path still available
- ✅ Lower maintenance burden

**The config system is now lean, focused, and enforces the nested-only policy cleanly.**
