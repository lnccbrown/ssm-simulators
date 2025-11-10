# Migration Guide: SimpleThetaProcessor → ModularThetaProcessor

## Overview

This guide helps you migrate from `SimpleThetaProcessor` to `ModularThetaProcessor` in `ssm-simulators`.

**Good news:** Most code needs NO changes! The `Simulator` class uses `ModularThetaProcessor` by default with full backward compatibility.

---

## Table of Contents

1. [Do You Need to Migrate?](#do-you-need-to-migrate)
2. [Quick Migration Checklist](#quick-migration-checklist)
3. [What Changed](#what-changed)
4. [Migration Scenarios](#migration-scenarios)
5. [Breaking Changes (None!)](#breaking-changes-none)
6. [Feature Comparison](#feature-comparison)
7. [Troubleshooting](#troubleshooting)

---

## Do You Need to Migrate?

### Check Your Code

**If you're using:**
```python
from ssms import Simulator

sim = Simulator("ddm")
result = sim.simulate(theta={...}, n_samples=1000)
```

✅ **No migration needed!** Your code automatically uses `ModularThetaProcessor`.

**If you're explicitly using:**
```python
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

processor = SimpleThetaProcessor()
# ...
```

⚠️ **Consider migrating** to benefit from new features, but it's not required.

---

## Quick Migration Checklist

- [ ] **Step 1:** Test your existing code (it should work as-is)
- [ ] **Step 2:** Remove explicit `SimpleThetaProcessor` usage (if any)
- [ ] **Step 3:** Add custom transformations (optional)
- [ ] **Step 4:** Run tests to verify equivalence

---

## What Changed

### Before (SimpleThetaProcessor)

**Architecture:**
- Monolithic 370-line `process_theta()` method
- Model-specific logic in long if-elif chains
- Hard to extend or customize
- No introspection

**Usage:**
```python
# Hardcoded in Simulator
sim = Simulator("lba2")  # Always used SimpleThetaProcessor

# Or explicit
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
processor = SimpleThetaProcessor()
theta_processed = processor.process_theta(theta, model_config, n_trials)
```

### After (ModularThetaProcessor)

**Architecture:**
- Modular pipeline of small, composable transformations
- Model-specific logic in registry mappings
- Easy to extend with custom transformations
- Full introspection support

**Usage:**
```python
# Automatic (default)
sim = Simulator("lba2")  # Automatically uses ModularThetaProcessor

# Explicit (if needed)
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
processor = ModularThetaProcessor()
theta_processed = processor.process_theta(theta, model_config, n_trials)

# With custom transformations
from ssms.basic_simulators.theta_transforms import SetDefaultValue
sim = Simulator("lba2", theta_transforms=[SetDefaultValue("custom", 42)])
```

---

## Migration Scenarios

### Scenario 1: Using Simulator Class (Most Common)

**Before:**
```python
from ssms import Simulator

sim = Simulator("ddm")
result = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
```

**After:**
```python
# SAME CODE! No changes needed.
from ssms import Simulator

sim = Simulator("ddm")
result = sim.simulate(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    n_samples=1000
)
```

**Status:** ✅ Works automatically

---

### Scenario 2: Explicit SimpleThetaProcessor Usage

**Before:**
```python
from ssms import Simulator
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

processor = SimpleThetaProcessor()
sim = Simulator("ddm", theta_processor=processor)
```

**After (Option A - Recommended):**
```python
# Just remove theta_processor parameter
from ssms import Simulator

sim = Simulator("ddm")  # Uses ModularThetaProcessor by default
```

**After (Option B - Keep Legacy):**
```python
# Keep using SimpleThetaProcessor if you have a specific reason
from ssms import Simulator
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

processor = SimpleThetaProcessor()
sim = Simulator("ddm", theta_processor=processor)  # Still works!
```

**Status:** ✅ Both work, Option A recommended

---

### Scenario 3: Custom Theta Processing

**Before:**
```python
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

class CustomProcessor(SimpleThetaProcessor):
    def process_theta(self, theta, model_config, n_trials):
        # Call parent
        theta = super().process_theta(theta, model_config, n_trials)
        
        # Add custom logic
        theta["my_custom_param"] = 42
        
        return theta

# Use it
processor = CustomProcessor()
sim = Simulator("ddm", theta_processor=processor)
```

**After (Option A - Custom Transformations):**
```python
from ssms import Simulator
from ssms.basic_simulators.theta_transforms import LambdaTransformation

# Define custom transformation
custom_transform = LambdaTransformation(
    func=lambda theta, cfg, n: theta.update({"my_custom_param": 42}) or theta,
    name="add_custom_param"
)

# Use it
sim = Simulator("ddm", theta_transforms=[custom_transform])
```

**After (Option B - Custom Processor):**
```python
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_transforms import ThetaProcessorRegistry, LambdaTransformation

# Create custom registry
registry = ThetaProcessorRegistry()

# Copy default transformations and add custom one
# (Or build from scratch)
registry.register_model("ddm", [
    # ... default transformations ...
    LambdaTransformation(
        lambda theta, cfg, n: theta.update({"my_custom_param": 42}) or theta,
        name="add_custom"
    )
])

processor = ModularThetaProcessor(registry=registry)
sim = Simulator("ddm", theta_processor=processor)
```

**Status:** ✅ Both work, Option A much simpler

---

### Scenario 4: Direct Processor Usage (No Simulator)

**Before:**
```python
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
from ssms.config import model_config

processor = SimpleThetaProcessor()
theta_processed = processor.process_theta(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    model_config=model_config['ddm'],
    n_trials=1
)
```

**After:**
```python
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.config import model_config

processor = ModularThetaProcessor()
theta_processed = processor.process_theta(
    theta={'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3},
    model_config=model_config['ddm'],
    n_trials=1
)
```

**Status:** ✅ Simple replacement

---

### Scenario 5: Testing/Validation

**Before:**
```python
def test_theta_processing():
    from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
    
    processor = SimpleThetaProcessor()
    theta = processor.process_theta({'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}, model_config, 1)
    
    assert 'v' in theta
    assert theta['v'].shape == (1,)
```

**After:**
```python
def test_theta_processing():
    from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
    
    processor = ModularThetaProcessor()
    theta = processor.process_theta({'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}, model_config, 1)
    
    assert 'v' in theta
    assert theta['v'].shape == (1,)
```

**Status:** ✅ Results are identical (verified by 233 equivalence tests)

---

## Breaking Changes (None!)

✅ **No breaking changes!**

The `ModularThetaProcessor` produces **identical results** to `SimpleThetaProcessor` for all 106 models, verified by comprehensive equivalence tests.

**What this means:**
- Your existing code works without changes
- Results are numerically identical
- Random seeds produce same sequences
- All tests should pass unchanged

---

## Feature Comparison

| Feature | SimpleThetaProcessor | ModularThetaProcessor |
|---------|---------------------|----------------------|
| **Basic theta processing** | ✅ | ✅ |
| **All 106 models supported** | ✅ | ✅ |
| **Backward compatible** | ✅ | ✅ |
| **Custom transformations** | ❌ Subclass required | ✅ Via parameter |
| **Introspection** | ❌ Black box | ✅ `describe()` method |
| **Modular architecture** | ❌ Monolithic | ✅ Composable |
| **Extensibility** | ⚠️ Hard | ✅ Easy |
| **Testing** | ⚠️ Black box testing | ✅ Per-transformation tests |
| **Performance** | Fast | Fast (minimal overhead) |
| **Maintenance** | ⚠️ 370-line method | ✅ Small components |

---

## Benefits of Migrating

### 1. **Custom Transformations**

**Before:** Had to subclass and override `process_theta()`
```python
class MyProcessor(SimpleThetaProcessor):
    def process_theta(self, theta, model_config, n_trials):
        theta = super().process_theta(theta, model_config, n_trials)
        # Add custom logic
        return theta
```

**After:** Just add transformations
```python
sim = Simulator("ddm", theta_transforms=[
    SetDefaultValue("my_param", 42)
])
```

### 2. **Introspection**

**Before:** No way to see what transformations are applied
```python
# What does this do? 🤷
processor.process_theta(theta, model_config, n_trials)
```

**After:** Full transparency
```python
sim = Simulator("lba2")
print(sim.theta_processor.registry.describe("lba2"))

# Output:
# Model: lba2
# Transformations (6):
#   1. LambdaTransformation(name='set_nact_2')
#   2. ColumnStackParameters(['v0', 'v1'] → 'v')
#   ...
```

### 3. **Easier Testing**

**Before:** Test entire 370-line method
```python
# Black box test
result = processor.process_theta(theta, config, n_trials)
assert result == expected  # Hard to debug if fails
```

**After:** Test individual transformations
```python
transform = SetDefaultValue("param", 42)
result = transform.apply(theta, config, n_trials)
assert result["param"] == 42  # Easy to debug
```

### 4. **Better Maintainability**

**Before:** 370-line if-elif chain
```python
if model == "lba2":
    # 10 lines
elif model == "lba3":
    # 10 lines
# ... 100 more models
```

**After:** Registry with small transformations
```python
registry.register_model("lba2", [
    SetDefaultValue("nact", 2),
    ColumnStackParameters(["v0", "v1"], "v"),
    ...
])
```

---

## Troubleshooting

### Issue 1: Different Results After Migration

**Symptom:** Results differ after switching to `ModularThetaProcessor`

**Diagnosis:**
```python
# Test both processors
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor

theta_in = {'v': 0.5, 'a': 1.0, 'z': 0.5, 't': 0.3}

old = SimpleThetaProcessor()
new = ModularThetaProcessor()

theta_old = old.process_theta(theta_in, model_config['ddm'], 1)
theta_new = new.process_theta(theta_in, model_config['ddm'], 1)

# Compare
for key in theta_old:
    if key in theta_new:
        print(f"{key}: {theta_old[key]} vs {theta_new[key]}")
```

**Solution:** 
- If you find differences, please report as a bug (should be identical)
- Workaround: Use `SimpleThetaProcessor` explicitly until resolved

### Issue 2: Custom Transforms Not Applied

**Symptom:** Custom transformations ignored

**Cause:** Using `SimpleThetaProcessor` explicitly

**Solution:**
```python
# Don't do this:
sim = Simulator("ddm", 
                theta_processor=SimpleThetaProcessor(),
                theta_transforms=[...])  # Ignored!

# Do this:
sim = Simulator("ddm", 
                theta_transforms=[...])  # Uses ModularThetaProcessor
```

### Issue 3: Import Errors

**Symptom:** `ModuleNotFoundError` or `ImportError`

**Solution:** Update imports:
```python
# Old
from ssms.basic_simulators.theta_processor import SimpleThetaProcessor

# New
from ssms.basic_simulators.modular_theta_processor import ModularThetaProcessor
from ssms.basic_simulators.theta_transforms import (
    ThetaTransformation,
    SetDefaultValue,
    # ... other transforms
)
```

### Issue 4: Tests Failing

**Symptom:** Tests pass with old processor, fail with new

**Diagnosis:**
1. Check if tests explicitly use `SimpleThetaProcessor`
2. Verify test expectations are correct
3. Compare processed theta dictionaries

**Solution:**
```python
# Update tests to use ModularThetaProcessor
def test_my_feature():
    # Old
    processor = SimpleThetaProcessor()
    
    # New
    processor = ModularThetaProcessor()
    
    # Or just use Simulator (uses ModularThetaProcessor by default)
    sim = Simulator("ddm")
```

---

## Migration Timeline

### Current (v0.7.x)

- ✅ Both processors supported
- ✅ `ModularThetaProcessor` is default
- ✅ `SimpleThetaProcessor` still available
- ✅ No deprecation warnings

### Future (v0.8.x)

- ⚠️ Deprecation warnings added to `SimpleThetaProcessor`
- ✅ Both processors still functional
- 📖 Migration guide prominently displayed

### Future (v1.0.x)

- ❌ `SimpleThetaProcessor` removed
- ✅ `ModularThetaProcessor` only option
- ✅ Full migration complete

**Recommendation:** Migrate now to avoid future deprecation warnings.

---

## FAQ

### Q: Do I have to migrate?

**A:** No, but recommended. `SimpleThetaProcessor` will be deprecated in future versions.

### Q: Will my tests break?

**A:** No. Both processors produce identical results (verified by 233 tests).

### Q: Can I use both processors?

**A:** Yes, you can use different processors for different `Simulator` instances.

### Q: How do I know which processor I'm using?

**A:** Check the `theta_processor` property:
```python
sim = Simulator("ddm")
print(type(sim.theta_processor))
# <class 'ModularThetaProcessor'> or <class 'SimpleThetaProcessor'>
```

### Q: What if I find a bug?

**A:** Please report it! Include:
1. Model name
2. Input theta
3. Expected vs actual output
4. Code to reproduce

### Q: Can I contribute transformations?

**A:** Yes! See [Custom Theta Transforms Guide](custom_theta_transforms_guide.md) and submit a PR.

---

## Summary

**Migration is easy:**
1. ✅ Most code needs NO changes
2. ✅ Results are identical
3. ✅ New features available immediately
4. ✅ Backward compatibility maintained

**Recommended action:**
```python
# Remove explicit SimpleThetaProcessor usage
# from ssms.basic_simulators.theta_processor import SimpleThetaProcessor  # Remove
# sim = Simulator("ddm", theta_processor=SimpleThetaProcessor())  # Remove

# Use default (ModularThetaProcessor automatically)
sim = Simulator("ddm")  # That's it!
```

**Need help?** See:
- [Theta Processor Guide](theta_processor_guide.md) - Full documentation
- [Custom Transforms Guide](custom_theta_transforms_guide.md) - Creating custom transformations
- [GitHub Issues](https://github.com/AlexanderFengler/ssm-simulators/issues) - Report problems

