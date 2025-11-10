# Modular ThetaProcessor Refactoring Plan

## Overview

Refactor the monolithic `SimpleThetaProcessor` (370+ lines, single method) into a modular, extensible transformation-based system. This refactoring will improve maintainability, testability, and extensibility while maintaining full backward compatibility.

## Goals

1. **Modularity**: Break 370-line method into composable transformation classes
2. **Extensibility**: Allow users to add custom transformations without modifying core code
3. **Testability**: Enable testing of individual transformations in isolation
4. **Integration**: Seamlessly integrate with the new `Simulator` class
5. **Backward Compatibility**: Maintain existing behavior during migration
6. **Documentation**: Provide clear examples and migration guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 ThetaTransformation (ABC)               │
│  - apply(theta, model_config, n_trials) → theta       │
└─────────────────────────────────────────────────────────┘
                        ▲
                        │ inherits
           ┌────────────┴────────────┐
           │                         │
  ┌────────────────────┐   ┌────────────────────┐
  │ SetDefaultValue    │   │ ColumnStackDrifts  │
  │ ExpandDimension    │   │ RenameParameter    │
  │ SetZeroArray       │   │ DeleteParameters   │
  │ ApplyMapping       │   │ ... custom ...     │
  └────────────────────┘   └────────────────────┘
           │                         │
           └────────────┬────────────┘
                        │ used by
           ┌────────────▼────────────┐
           │ ThetaProcessorRegistry  │
           │  - register_model()     │
           │  - register_family()    │
           │  - get_processor()      │
           └────────────┬────────────┘
                        │ used by
           ┌────────────▼────────────┐
           │ ModularThetaProcessor   │
           │  - process_theta()      │
           └─────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Core Infrastructure)

**Goal**: Create base classes and infrastructure without breaking existing code.

**Files to Create**:
1. `ssms/basic_simulators/theta_transforms/__init__.py`
2. `ssms/basic_simulators/theta_transforms/base.py`
3. `ssms/basic_simulators/theta_transforms/common.py`
4. `ssms/basic_simulators/theta_transforms/registry.py`
5. `ssms/basic_simulators/modular_theta_processor.py`

**Components**:

#### 1.1 Base Transformation Class
```python
# theta_transforms/base.py
class ThetaTransformation(ABC):
    """Base class for theta parameter transformations."""
    
    @abstractmethod
    def apply(self, theta: dict, model_config: dict, n_trials: int) -> dict:
        """Apply transformation to theta parameters."""
        pass
    
    def __repr__(self):
        """String representation for debugging."""
        pass
```

#### 1.2 Common Transformations
```python
# theta_transforms/common.py
- SetDefaultValue
- ExpandDimension
- ColumnStackParameters
- RenameParameter
- DeleteParameters
- SetZeroArray
- TileArray
- ApplyMapping
- ConditionalTransform
```

#### 1.3 Registry System
```python
# theta_transforms/registry.py
class ThetaProcessorRegistry:
    - register_model(name, transforms)
    - register_family(family_name, matcher, transforms)
    - get_processor(model_name) → list[ThetaTransformation]
    - list_registered_models() → list[str]
    - list_registered_families() → list[str]
```

#### 1.4 Modular Processor
```python
# modular_theta_processor.py
class ModularThetaProcessor(AbstractThetaProcessor):
    - __init__(registry)
    - process_theta(theta, config, n_trials)
    - _build_default_registry() → registry with all models
```

**Acceptance Criteria**:
- All base classes and common transformations implemented
- Registry supports model and family registration
- Can create and apply transformations in sequence
- No integration with Simulator yet (pure library code)

---

### Phase 2: Migration (Populate Registry)

**Goal**: Translate all existing `SimpleThetaProcessor` logic into registry entries.

**Tasks**:

#### 2.1 Analyze Current Implementation
Map all model-specific logic blocks to transformations:
```
Lines 92-109:  No-op models (DDM, angle, etc.)
Lines 112-113: Dynamic drift models → SetZeroArray("v")
Lines 115-118: Dual drift models → SetZeroArray("v", shape=(n, 2))
Lines 120-126: ddm_st → ApplyMapping transformations
Lines 169-178: lba2 → 5 transformations
... etc for all 40+ model patterns
```

#### 2.2 Create Model-Specific Transforms
For models needing unique logic:
```python
# theta_transforms/model_specific.py
- LBATransform (for LBA models)
- RaceTransform (for race models)
- Mic2Transform (for MIC2 variants)
- RandomVariableTransform (for _rv models)
```

#### 2.3 Populate Default Registry
Implement `_build_default_registry()` with all 40+ models:
```python
def _build_default_registry():
    registry = ThetaProcessorRegistry()
    
    # Single-particle models (no transforms)
    for model in ["ddm", "angle", "weibull", ...]:
        registry.register_model(model, [])
    
    # LBA models
    registry.register_model("lba2", [...])
    registry.register_model("lba3", [...])
    
    # Race model families
    registry.register_family("race_2", matcher, [...])
    registry.register_family("race_3", matcher, [...])
    registry.register_family("race_4", matcher, [...])
    
    # LCA model families
    # Sequential/Parallel model families
    # MIC2 model families
    # ... etc
    
    return registry
```

**Acceptance Criteria**:
- All models from `SimpleThetaProcessor` have registry entries
- Family matchers correctly identify model variants
- Registry lookups return correct transformations
- Transformation application produces identical results to `SimpleThetaProcessor`

---

### Phase 3: Testing (Verification)

**Goal**: Ensure new system is equivalent to old system.

**Files to Create**:
1. `tests/test_theta_transforms.py`
2. `tests/test_theta_registry.py`
3. `tests/test_modular_theta_processor.py`
4. `tests/test_theta_processor_equivalence.py`

**Test Categories**:

#### 3.1 Unit Tests (Individual Transformations)
```python
def test_expand_dimension():
    theta = {"a": np.array([1.0, 2.0])}
    transform = ExpandDimension(["a"])
    result = transform.apply(theta, {}, 2)
    assert result["a"].shape == (2, 1)

def test_column_stack_drifts():
    theta = {"v0": [0.5], "v1": [0.6], "v2": [0.7]}
    transform = ColumnStackParameters(["v0", "v1", "v2"], "v")
    result = transform.apply(theta, {}, 1)
    assert result["v"].shape == (1, 3)
    assert "v0" not in result  # Original keys removed
```

#### 3.2 Integration Tests (Full Pipelines)
```python
def test_lba3_pipeline():
    theta = {"v0": [0.5], "v1": [0.6], "v2": [0.7], "A": [0.5], "b": [1.0]}
    transforms = registry.get_processor("lba3")
    for transform in transforms:
        theta = transform.apply(theta, config, 1)
    assert "v" in theta
    assert theta["v"].shape == (1, 3)
    assert "nact" in theta
```

#### 3.3 Equivalence Tests (Old vs New)
```python
@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
def test_processor_equivalence(model_name):
    """Verify ModularThetaProcessor produces identical results to SimpleThetaProcessor."""
    theta_input = generate_test_theta(model_name)
    config = model_config[model_name]
    
    # Old processor
    old_result = SimpleThetaProcessor().process_theta(
        deepcopy(theta_input), config, n_trials=10
    )
    
    # New processor
    new_result = ModularThetaProcessor().process_theta(
        deepcopy(theta_input), config, n_trials=10
    )
    
    # Should be identical
    assert_theta_equal(old_result, new_result)
```

**Acceptance Criteria**:
- 100+ unit tests covering all transformation classes
- Integration tests for each model family
- Equivalence tests pass for all 40+ models
- Edge cases handled (missing params, wrong shapes, etc.)

---

### Phase 4: Integration (Simulator Class)

**Goal**: Integrate modular processor with `Simulator` class.

**Files to Modify**:
1. `ssms/basic_simulators/simulator_class.py`
2. `ssms/config/config_builder.py`

**Changes**:

#### 4.1 Simulator Class Updates
```python
class Simulator:
    def __init__(
        self,
        model: str | dict | None = None,
        boundary: str | Callable | None = None,
        drift: str | Callable | None = None,
        simulator_function: Callable | None = None,
        theta_processor: ThetaProcessor | None = None,  # NEW
        theta_transforms: list[ThetaTransformation] | None = None,  # NEW
        **config_overrides,
    ):
        # Set processor (default to modular with registry)
        self._theta_processor = theta_processor or ModularThetaProcessor()
        self._custom_transforms = theta_transforms or []
        
        # Build config with transforms
        self._config = self._build_config(...)
    
    def _build_config(self, ...):
        # Get default transformations for model
        model_name = config.get("name", "")
        default_transforms = self._theta_processor.registry.get_processor(model_name)
        
        # Combine with custom transforms
        config["theta_transforms"] = default_transforms + self._custom_transforms
        
        return config
    
    def simulate(self, theta, ...):
        # Use configured processor
        theta = self._theta_processor.process_theta(theta, self._config, n_trials)
        ...
```

#### 4.2 ConfigBuilder Updates
```python
class ConfigBuilder:
    @staticmethod
    def add_theta_transforms(config, transforms):
        """Add theta transformations to configuration."""
        ...
    
    @staticmethod
    def from_model(model_name, **overrides):
        """Include theta transforms in config."""
        config = deepcopy(model_config[model_name])
        
        # Add default transforms from registry
        registry = get_default_registry()
        config["theta_transforms"] = registry.get_processor(model_name)
        
        config.update(overrides)
        return config
```

#### 4.3 Backward Compatibility Flag
```python
class Simulator:
    def __init__(
        self,
        ...,
        use_legacy_theta_processor: bool = False,  # Temporary migration flag
        ...
    ):
        if use_legacy_theta_processor:
            warnings.warn(
                "SimpleThetaProcessor is deprecated. Use ModularThetaProcessor.",
                DeprecationWarning
            )
            self._theta_processor = SimpleThetaProcessor()
        else:
            self._theta_processor = theta_processor or ModularThetaProcessor()
```

**Acceptance Criteria**:
- `Simulator` accepts `theta_processor` and `theta_transforms` parameters
- Default behavior uses `ModularThetaProcessor`
- Custom transformations can be added to built-in models
- `ConfigBuilder` includes theta transforms
- All existing `Simulator` tests still pass

---

### Phase 5: Documentation

**Goal**: Comprehensive documentation for users and developers.

**Files to Create**:
1. `docs/theta_processor_guide.md`
2. `docs/custom_theta_transforms.md`
3. `docs/migration_theta_processor.md`

**Content**:

#### 5.1 User Guide
- What are theta transformations?
- How to use custom transformations with `Simulator`
- Examples for common use cases
- Available transformation classes
- How to create custom transformations

#### 5.2 Developer Guide
- Architecture overview
- How to add new transformation types
- Registry mechanism explained
- How to register new model families
- Testing guidelines

#### 5.3 Migration Guide
- Differences between old and new processor
- How to migrate custom processor implementations
- Deprecation timeline
- Troubleshooting common issues

**Acceptance Criteria**:
- All public APIs documented with docstrings
- User guide with 10+ examples
- Migration guide for existing users
- API reference for all transformation classes

---

### Phase 6: Deprecation & Cleanup

**Goal**: Phase out `SimpleThetaProcessor` gracefully.

**Timeline**:
- v0.8.0: Introduce `ModularThetaProcessor`, keep both
- v0.9.0: Deprecate `SimpleThetaProcessor` (warnings)
- v1.0.0: Remove `SimpleThetaProcessor`

**Tasks**:

#### 6.1 Add Deprecation Warnings
```python
class SimpleThetaProcessor(AbstractThetaProcessor):
    def __init__(self):
        warnings.warn(
            "SimpleThetaProcessor is deprecated and will be removed in v1.0.0. "
            "Use ModularThetaProcessor instead.",
            DeprecationWarning,
            stacklevel=2
        )
```

#### 6.2 Update Default in Simulator
```python
# v0.9.0: Change default but allow override
def __init__(self, ..., theta_processor=None, ...):
    # Default to modular processor
    self._theta_processor = theta_processor or ModularThetaProcessor()
```

#### 6.3 Remove Legacy Code (v1.0.0)
- Delete `SimpleThetaProcessor` class
- Remove `use_legacy_theta_processor` flag
- Clean up tests

**Acceptance Criteria**:
- Deprecation warnings added with clear messages
- Documentation updated with migration timeline
- Legacy code removal tracked in issues
- Breaking change documented in CHANGELOG

---

## File Structure

```
ssms/
├── basic_simulators/
│   ├── __init__.py
│   ├── simulator.py (uses SimpleThetaProcessor - legacy)
│   ├── simulator_class.py (uses ModularThetaProcessor - new)
│   ├── theta_processor.py (AbstractThetaProcessor + SimpleThetaProcessor)
│   ├── modular_theta_processor.py (NEW - ModularThetaProcessor)
│   └── theta_transforms/ (NEW)
│       ├── __init__.py
│       ├── base.py (ThetaTransformation ABC)
│       ├── common.py (common transformations)
│       ├── model_specific.py (model-specific transforms)
│       └── registry.py (ThetaProcessorRegistry)
│
├── config/
│   ├── __init__.py
│   └── config_builder.py (updated with theta_transforms support)
│
tests/
├── test_theta_transforms.py (NEW - unit tests)
├── test_theta_registry.py (NEW - registry tests)
├── test_modular_theta_processor.py (NEW - integration tests)
├── test_theta_processor_equivalence.py (NEW - equivalence tests)
└── test_simulator_class.py (updated with theta transform tests)

docs/
├── theta_processor_guide.md (NEW)
├── custom_theta_transforms.md (NEW)
└── migration_theta_processor.md (NEW)
```

## Success Criteria

- [ ] All 40+ models from `SimpleThetaProcessor` migrated to registry
- [ ] 100+ tests with 100% pass rate
- [ ] Equivalence tests confirm identical behavior
- [ ] `Simulator` class fully integrated
- [ ] Zero breaking changes for existing users
- [ ] Comprehensive documentation
- [ ] Performance: No significant regression (< 5% slower acceptable)
- [ ] Code reduction: 370-line method → 10-20 line transformation classes

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Behavioral differences between old/new | High | Extensive equivalence testing |
| Performance regression | Medium | Benchmark suite, optimize hot paths |
| Breaking changes for users | High | Maintain SimpleThetaProcessor during migration |
| Incomplete model coverage | Medium | Systematic analysis of all 464 lines |
| Complex family matching | Low | Clear documentation, regex support |

## Open Questions

1. Should we support regex matchers or stick with simple functions?
2. How to handle transformations that depend on other transformations?
3. Should transformations be stateless or allow state?
4. Cache transformed results for same theta?
5. Parallel transformation application?

## Timeline Estimate

- Phase 1 (Foundation): 2-3 days
- Phase 2 (Migration): 3-4 days
- Phase 3 (Testing): 2-3 days
- Phase 4 (Integration): 1-2 days
- Phase 5 (Documentation): 1-2 days
- Phase 6 (Deprecation): Spread over multiple releases

**Total**: ~10-15 days of focused development

## Next Steps

1. Review and iterate on this plan
2. Start with Phase 1: Create base infrastructure
3. Build one example model family (e.g., LBA models) end-to-end
4. Validate approach before migrating all models
5. Iterate based on learnings

