# Parameter Transform & Theta Processor Renaming Plan

## Decision

Rename for clarity and semantic accuracy:

```python
ParameterTransform      → ParameterSamplingConstraint
ThetaProcessor          → ParameterSimulatorAdapter
```

## Rationale

### ParameterSamplingConstraint
- ✅ Describes **what** it does: constrains parameters
- ✅ Describes **when** it runs: during sampling phase
- ✅ Captures semantic purpose: validation and constraint enforcement
- ✅ Indicates it can trigger resampling

### ParameterSimulatorAdapter
- ✅ Describes **what** it does: adapts parameter format
- ✅ Describes **for what**: simulator interface
- ✅ Uses established design pattern (Adapter Pattern)
- ✅ Emphasizes bridging role between parameter format and simulator expectations
- ✅ Removes confusing "Theta" terminology

---

## Scope of Changes

### Phase 1: Core Infrastructure

#### 1.1 Protocol Definitions
**File:** `ssms/dataset_generators/parameter_samplers/protocols.py`

```python
# OLD
class ParameterTransformProtocol(Protocol):
    def apply(self, theta: dict) -> dict: ...

# NEW
class ParameterSamplingConstraintProtocol(Protocol):
    def apply(self, theta: dict) -> dict: ...
```

**Actions:**
- Rename `ParameterTransformProtocol` → `ParameterSamplingConstraintProtocol`
- Add deprecated alias for backward compatibility
- Update all docstrings

---

**File:** `ssms/basic_simulators/theta_processor.py`

```python
# OLD
class AbstractThetaProcessor(ABC):
    @abstractmethod
    def process_theta(self, theta, model_config, n_trials): ...

class SimpleThetaProcessor(AbstractThetaProcessor):
    ...

# NEW (rename file to parameter_simulator_adapter.py)
class AbstractParameterSimulatorAdapter(ABC):
    @abstractmethod
    def adapt_parameters(self, theta, model_config, n_trials): ...

class SimpleParameterSimulatorAdapter(AbstractParameterSimulatorAdapter):
    ...
```

**Actions:**
- Rename file: `theta_processor.py` → `parameter_simulator_adapter.py`
- Rename classes: `AbstractThetaProcessor` → `AbstractParameterSimulatorAdapter`
- Rename classes: `SimpleThetaProcessor` → `SimpleParameterSimulatorAdapter`
- Rename method: `process_theta` → `adapt_parameters` (or keep as `adapt` for brevity)
- Add deprecated aliases
- Update all docstrings

---

**File:** `ssms/basic_simulators/modular_theta_processor.py`

```python
# OLD
class ModularThetaProcessor(AbstractThetaProcessor):
    ...

# NEW (rename file to modular_parameter_simulator_adapter.py)
class ModularParameterSimulatorAdapter(AbstractParameterSimulatorAdapter):
    ...
```

**Actions:**
- Rename file: `modular_theta_processor.py` → `modular_parameter_simulator_adapter.py`
- Rename class: `ModularThetaProcessor` → `ModularParameterSimulatorAdapter`
- Update all references to registry
- Add deprecated alias

---

#### 1.2 Concrete Implementations

**Directory:** `ssms/dataset_generators/parameter_samplers/transforms/`

```python
# Rename directory to: constraints/
```

**Files to rename:**
- `transforms/swap.py` → `constraints/swap.py`
- `transforms/normalize.py` → `constraints/normalize.py`
- `transforms/adapters.py` → `constraints/adapters.py`
- `transforms/registry.py` → `constraints/registry.py`
- `transforms/factory.py` → `constraints/factory.py`
- `transforms/__init__.py` → `constraints/__init__.py`

**Classes to rename:**
- `SwapIfLessTransform` → `SwapIfLessConstraint`
- `NormalizeToSumTransform` → `NormalizeToSumConstraint`
- `FunctionTransformAdapter` → `FunctionConstraintAdapter`

**Functions to rename:**
- `create_transform_from_config` → `create_constraint_from_config`
- `get_transforms_from_model_config` → `get_constraints_from_model_config`
- `register_transform_function` → `register_constraint_function`
- `register_transform_class` → `register_constraint_class`

---

**Directory:** `ssms/basic_simulators/theta_transforms/`

```python
# Rename directory to: parameter_adapters/
```

**Files to rename:**
- `theta_transforms/base.py` → `parameter_adapters/base.py`
- `theta_transforms/common.py` → `parameter_adapters/common.py`
- `theta_transforms/registry.py` → `parameter_adapters/registry.py`
- `theta_transforms/__init__.py` → `parameter_adapters/__init__.py`

**Classes to rename:**
- `ThetaTransformation` → `ParameterAdaptation`
- `ThetaProcessorRegistry` → `ParameterAdapterRegistry`
- `SetDefaultValue` → (keep as is, or `SetDefaultParameter`)
- `ExpandDimension` → (keep as is, or `ExpandParameterDimension`)
- `RenameParameter` → (keep as is)
- `ColumnStackParameters` → (keep as is)
- `DeleteParameters` → (keep as is)
- `LambdaTransformation` → `LambdaAdaptation`

---

### Phase 2: Integration Points

#### 2.1 Simulator Class

**File:** `ssms/basic_simulators/simulator_class.py`

```python
# OLD
def __init__(
    self,
    model: str | dict | None = None,
    theta_processor: AbstractThetaProcessor | None = None,
    theta_transforms: list[ThetaTransformation] | None = None,
    **config_overrides,
):
    self._theta_processor = theta_processor or ModularThetaProcessor()
    ...

# NEW
def __init__(
    self,
    model: str | dict | None = None,
    parameter_adapter: AbstractParameterSimulatorAdapter | None = None,
    parameter_adaptations: list[ParameterAdaptation] | None = None,
    **config_overrides,
):
    self._parameter_adapter = parameter_adapter or ModularParameterSimulatorAdapter()
    ...
```

**Actions:**
- Rename parameter: `theta_processor` → `parameter_adapter`
- Rename parameter: `theta_transforms` → `parameter_adaptations`
- Rename attribute: `self._theta_processor` → `self._parameter_adapter`
- Update all method calls
- Add deprecation warnings for old parameters

---

**File:** `ssms/basic_simulators/simulator.py`

```python
# OLD
theta = SimpleThetaProcessor().process_theta(theta, model_config_local, n_trials)

# NEW
theta = SimpleParameterSimulatorAdapter().adapt_parameters(theta, model_config_local, n_trials)
```

**Actions:**
- Update function-based simulator to use new names
- Keep backward compatibility

---

#### 2.2 Parameter Samplers

**File:** `ssms/dataset_generators/parameter_samplers/base_sampler.py`

```python
# OLD
def __init__(
    self,
    param_space: dict[str, tuple[float, float]],
    transforms: list[ParameterTransformProtocol] | None = None,
):
    self.transforms = transforms or []
    ...

# NEW
def __init__(
    self,
    param_space: dict[str, tuple[float, float]],
    constraints: list[ParameterSamplingConstraintProtocol] | None = None,
):
    self.constraints = constraints or []
    ...
```

**Actions:**
- Rename parameter: `transforms` → `constraints`
- Rename attribute: `self.transforms` → `self.constraints`
- Update loop: `for transform in self.transforms:` → `for constraint in self.constraints:`
- Update method calls: `transform.apply()` → `constraint.apply()`
- Update all docstrings

---

#### 2.3 Pipeline Classes

**File:** `ssms/dataset_generators/pipelines/simulation_pipeline.py`

```python
# OLD
from ssms.dataset_generators.parameter_samplers.transforms.factory import (
    get_transforms_from_model_config,
)

transforms = get_transforms_from_model_config(self.model_config)
self._param_sampler = UniformParameterSampler(
    param_space=param_bounds_dict,
    transforms=transforms,
)

# NEW
from ssms.dataset_generators.parameter_samplers.constraints.factory import (
    get_constraints_from_model_config,
)

constraints = get_constraints_from_model_config(self.model_config)
self._param_sampler = UniformParameterSampler(
    param_space=param_bounds_dict,
    constraints=constraints,
)
```

**Actions:**
- Update imports
- Rename local variables
- Update all references

---

**File:** `ssms/dataset_generators/pipelines/pyddm_pipeline.py`

Same changes as `simulation_pipeline.py`

---

### Phase 3: Model Configuration

#### 3.1 Model Config Field Names

**File:** `ssms/config/_modelconfig/`

```python
# OLD
model_config = {
    "parameter_transforms": [
        {"type": "swap", "param_a": "a", "param_b": "z"},
    ]
}

# NEW (keep old name for backward compatibility)
model_config = {
    "parameter_sampling_constraints": [  # New preferred name
        {"type": "swap", "param_a": "a", "param_b": "z"},
    ],
    # "parameter_transforms": [...],  # Still supported (deprecated)
}
```

**Actions:**
- Support both `parameter_transforms` and `parameter_sampling_constraints`
- Log deprecation warning when old name is used
- Update all built-in model configs to use new name
- Update factory to check both field names

---

### Phase 4: Tests

#### 4.1 Test Files to Rename

```
tests/dataset_generators/parameter_samplers/test_transforms.py
  → test_constraints.py

tests/dataset_generators/parameter_samplers/test_transform_factory.py
  → test_constraint_factory.py

tests/dataset_generators/parameter_samplers/test_custom_transform_integration.py
  → test_custom_constraint_integration.py

tests/test_theta_processor_equivalence.py
  → test_parameter_adapter_equivalence.py

tests/test_simulator_theta_processor_integration.py
  → test_simulator_parameter_adapter_integration.py

tests/test_theta_transforms_basic.py
  → test_parameter_adaptations_basic.py
```

#### 4.2 Test Updates

**All test files need:**
- Import statement updates
- Class name updates
- Variable name updates
- Assertion message updates

---

### Phase 5: Documentation

#### 5.1 Documentation Files to Rename/Update

```
docs/custom_transforms.md
  → docs/custom_parameter_sampling_constraints.md

docs/theta_processor_guide.md
  → docs/parameter_simulator_adapter_guide.md

docs/migration_guide_theta_processor.md
  → docs/migration_guide_parameter_adapters.md

docs/custom_theta_transforms_guide.md
  → docs/custom_parameter_adaptations_guide.md

docs/PARAMETER_TRANSFORMS_VS_THETA_PROCESSORS.md
  → docs/PARAMETER_SAMPLING_CONSTRAINTS_VS_SIMULATOR_ADAPTERS.md
```

#### 5.2 New Migration Guide

Create `docs/NAMING_MIGRATION_GUIDE.md` to help users transition:

```markdown
# Naming Migration Guide

## Quick Reference

| Old Name | New Name | Where Used |
|----------|----------|------------|
| ParameterTransform | ParameterSamplingConstraint | Sampling phase |
| ThetaProcessor | ParameterSimulatorAdapter | Simulation phase |
| parameter_transforms | parameter_sampling_constraints | Model config |
| theta_processor | parameter_adapter | Simulator init |
| ...
```

---

### Phase 6: Examples

#### 6.1 Example Files to Update

```
examples/custom_transforms_example.py
  → examples/custom_constraints_example.py
```

**Update all code examples to use new names**

---

### Phase 7: Tutorials

#### 7.1 Tutorial Updates

**File:** `notebooks/tutorial_05_custom_models.ipynb`

- Section 4: "Parameter Transforms" → "Parameter Sampling Constraints"
- Update all code cells with new imports
- Update all explanations
- Update terminology throughout

---

## Backward Compatibility Strategy

### Deprecated Aliases (Keep for 1-2 versions)

**File:** `ssms/dataset_generators/parameter_samplers/protocols.py`

```python
# New names (preferred)
class ParameterSamplingConstraintProtocol(Protocol):
    """Protocol for parameter sampling constraints."""
    ...

# Deprecated aliases
ParameterTransformProtocol = ParameterSamplingConstraintProtocol  # Deprecated

def __getattr__(name):
    if name == "ParameterTransformProtocol":
        warnings.warn(
            "ParameterTransformProtocol is deprecated, use ParameterSamplingConstraintProtocol",
            DeprecationWarning,
            stacklevel=2,
        )
        return ParameterSamplingConstraintProtocol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Dual Field Support in Model Config

**File:** `ssms/dataset_generators/parameter_samplers/constraints/factory.py`

```python
def get_constraints_from_model_config(model_config: dict) -> list:
    """Get constraints from model config.

    Supports both old and new field names:
    - "parameter_sampling_constraints" (new, preferred)
    - "parameter_transforms" (old, deprecated)
    """
    # Try new name first
    if "parameter_sampling_constraints" in model_config:
        return [create_constraint_from_config(cfg)
                for cfg in model_config["parameter_sampling_constraints"]]

    # Fall back to old name (with warning)
    if "parameter_transforms" in model_config:
        warnings.warn(
            "'parameter_transforms' is deprecated, use 'parameter_sampling_constraints'",
            DeprecationWarning,
            stacklevel=2,
        )
        return [create_constraint_from_config(cfg)
                for cfg in model_config["parameter_transforms"]]

    return []
```

### Dual Parameter Support in Simulator

**File:** `ssms/basic_simulators/simulator_class.py`

```python
def __init__(
    self,
    model: str | dict | None = None,
    parameter_adapter: AbstractParameterSimulatorAdapter | None = None,
    theta_processor: AbstractParameterSimulatorAdapter | None = None,  # Deprecated
    parameter_adaptations: list[ParameterAdaptation] | None = None,
    theta_transforms: list[ParameterAdaptation] | None = None,  # Deprecated
    **config_overrides,
):
    # Handle deprecated parameters
    if theta_processor is not None:
        warnings.warn(
            "'theta_processor' is deprecated, use 'parameter_adapter'",
            DeprecationWarning,
            stacklevel=2,
        )
        parameter_adapter = parameter_adapter or theta_processor

    if theta_transforms is not None:
        warnings.warn(
            "'theta_transforms' is deprecated, use 'parameter_adaptations'",
            DeprecationWarning,
            stacklevel=2,
        )
        parameter_adaptations = parameter_adaptations or theta_transforms

    # Use new names internally
    self._parameter_adapter = parameter_adapter or ModularParameterSimulatorAdapter()
    self._custom_adaptations = parameter_adaptations or []
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
1. ✅ Update protocols
2. ✅ Rename core classes
3. ✅ Add deprecated aliases
4. ✅ Update imports in __init__ files

### Phase 2: Integration (Week 1-2)
1. ✅ Update Simulator class
2. ✅ Update parameter samplers
3. ✅ Update pipeline classes
4. ✅ Add dual parameter support

### Phase 3: Tests (Week 2)
1. ✅ Update all test files
2. ✅ Verify backward compatibility tests
3. ✅ Add deprecation warning tests

### Phase 4: Documentation (Week 2-3)
1. ✅ Update all guides
2. ✅ Create migration guide
3. ✅ Update tutorials
4. ✅ Update examples

### Phase 5: Cleanup (Week 3)
1. ✅ Run all tests
2. ✅ Update CHANGELOG
3. ✅ Final review

### Phase 6: Deprecation (Future version)
1. ⏳ Remove deprecated aliases (1-2 versions later)
2. ⏳ Remove old field names
3. ⏳ Update deprecation notices

---

## Testing Strategy

### Unit Tests
- Test new names work correctly
- Test old names still work (with warnings)
- Test deprecated aliases
- Test dual field support in configs

### Integration Tests
- Test full pipeline with new names
- Test full pipeline with old names (backward compat)
- Test mixed usage (some old, some new)

### Deprecation Tests
```python
def test_deprecated_transform_protocol():
    with pytest.warns(DeprecationWarning, match="ParameterTransformProtocol is deprecated"):
        protocol = ParameterTransformProtocol  # Old name
```

---

## File Checklist

### Core Files
- [ ] `ssms/dataset_generators/parameter_samplers/protocols.py`
- [ ] `ssms/basic_simulators/theta_processor.py` → `parameter_simulator_adapter.py`
- [ ] `ssms/basic_simulators/modular_theta_processor.py` → `modular_parameter_simulator_adapter.py`
- [ ] `ssms/dataset_generators/parameter_samplers/transforms/` → `constraints/`
- [ ] `ssms/basic_simulators/theta_transforms/` → `parameter_adapters/`

### Integration Files
- [ ] `ssms/basic_simulators/simulator_class.py`
- [ ] `ssms/basic_simulators/simulator.py`
- [ ] `ssms/dataset_generators/parameter_samplers/base_sampler.py`
- [ ] `ssms/dataset_generators/pipelines/simulation_pipeline.py`
- [ ] `ssms/dataset_generators/pipelines/pyddm_pipeline.py`

### Test Files
- [ ] All test files in `tests/dataset_generators/parameter_samplers/`
- [ ] All test files in `tests/` related to theta processors
- [ ] Integration tests

### Documentation Files
- [ ] All docs in `docs/` directory
- [ ] Tutorial notebooks
- [ ] Examples
- [ ] README files

### Configuration Files
- [ ] All model configs with `parameter_transforms` field
- [ ] Factory functions that read config fields

---

## Rollout Plan

### Version N (Current)
- Implement all renamings
- Add deprecated aliases
- Add deprecation warnings
- Update all documentation
- Keep backward compatibility 100%

### Version N+1
- Keep deprecated aliases
- Deprecation warnings remain
- Remove from new documentation (focus on new names)

### Version N+2
- Remove deprecated aliases
- Remove dual field support
- Breaking change (major version bump)

---

## Communication

### Changelog Entry

```markdown
## [Version] - Date

### Changed - BREAKING (in future version)
- **Renamed for clarity**: `ParameterTransform` → `ParameterSamplingConstraint`
- **Renamed for clarity**: `ThetaProcessor` → `ParameterSimulatorAdapter`
- Old names still work but are deprecated and will be removed in version N+2
- See migration guide: `docs/NAMING_MIGRATION_GUIDE.md`

### Deprecated
- `ParameterTransformProtocol` - use `ParameterSamplingConstraintProtocol`
- `AbstractThetaProcessor` - use `AbstractParameterSimulatorAdapter`
- `SimpleThetaProcessor` - use `SimpleParameterSimulatorAdapter`
- `ModularThetaProcessor` - use `ModularParameterSimulatorAdapter`
- Config field `parameter_transforms` - use `parameter_sampling_constraints`
- Simulator parameter `theta_processor` - use `parameter_adapter`
- Simulator parameter `theta_transforms` - use `parameter_adaptations`
```

### User Communication

Create announcement in:
- README
- GitHub Release Notes
- Documentation homepage
- Migration guide

---

## Risk Assessment

### Low Risk
- Adding new names alongside old ones
- Deprecation warnings
- Documentation updates

### Medium Risk
- Test coverage for all renamed components
- Ensuring all import paths are updated
- Tutorial notebook updates

### High Risk
- Missing a reference to old names in third-party code
- Breaking changes for users who depend on old names

### Mitigation
- Comprehensive deprecation period (2+ versions)
- Clear migration guide
- Automated tests for backward compatibility
- Search codebase thoroughly for all references

---

## Success Criteria

- [ ] All new names implemented
- [ ] All old names work with deprecation warnings
- [ ] 100% test coverage for both old and new names
- [ ] All documentation updated
- [ ] Migration guide complete
- [ ] All tutorials updated
- [ ] CHANGELOG updated
- [ ] No test failures
- [ ] Performance unchanged

---

## Next Steps

1. Review and approve this plan
2. Create implementation branch
3. Implement Phase 1 (Foundation)
4. Run tests after each phase
5. Peer review
6. Merge to main
7. Release with clear migration guide

---

**Estimated Effort**: 2-3 weeks for complete implementation and testing
**Backward Compatibility**: 2 versions (N+1, N+2 will break)
**Documentation Effort**: ~40% of total time
