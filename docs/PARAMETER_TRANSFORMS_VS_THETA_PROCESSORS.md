# Parameter Transforms vs Theta Processors: Architectural Analysis

## Executive Summary

**Should we unify parameter transforms and theta processors?**

**Recommendation: NO - Keep them separate.**

The separation between parameter transforms and theta processors reflects a fundamental architectural distinction between **data generation** and **simulation execution** phases. While they may seem redundant at first glance, they serve different purposes at different execution points with different performance and semantic requirements.

---

## Current Architecture

### Parameter Transforms
- **Location**: `ssms/dataset_generators/parameter_samplers/`
- **Applied**: During data generation (sampling phase)
- **When**: Once per parameter set in `ParameterSampler.sample()` (line 112-114 in `base_sampler.py`)
- **Frequency**: ~100-1000 times per training data generation run
- **Context**: No access to `model_config`, `n_trials`, or simulation context
- **Purpose**:
  - Enforce mathematical constraints between parameters (e.g., `a > z`)
  - Apply sampling-space transformations (e.g., sample in log-space, transform to real-space)
  - Normalize or scale sampled values
  - **Can trigger resampling** if validation fails

**Examples**:
- `SwapIfLessTransform(param_a="a", param_b="z")` - Ensure boundary > starting point
- `NormalizeToSumTransform(["v1", "v2", "v3"])` - Probabilities sum to 1
- `exponential_v(theta)` - Transform `v` from log-space to real-space

### Theta Processors
- **Location**: `ssms/basic_simulators/`
- **Applied**: During simulation (execution phase)
- **When**: Every call to `Simulator.simulate()` (line 622 in `simulator_class.py`)
- **Frequency**: Could be 1000s of times per parameter set (for training data generation with multiple subdatasets)
- **Context**: Has access to `model_config`, `n_trials`, and full simulation context
- **Purpose**:
  - Prepare parameters for simulator consumption (array dimensions)
  - Set model-specific defaults
  - Handle parameter renaming (e.g., `A → z`, `b → a` for LBA)
  - Stack parameters (e.g., `v0, v1 → v`)
  - Expand to match `n_trials`

**Examples**:
- `ExpandDimension(["v", "a"])` - Shape (1,) → (1, 1) for trial-by-trial variation
- `SetDefaultValue("t", np.zeros(n_trials))` - Default non-decision time
- `RenameParameter("A", "z", expand_dims=True)` - LBA parameter convention
- `ColumnStackParameters(["v0", "v1"], "v")` - Combine drift rates

---

## Key Differences

| Aspect | Parameter Transforms | Theta Processors |
|--------|---------------------|------------------|
| **Phase** | Data Generation | Simulation Execution |
| **Execution Point** | After sampling, before simulation | Right before simulator call |
| **Frequency** | Once per parameter set | Once per `simulate()` call |
| **Context** | Sampling-only (no `model_config`, `n_trials`) | Full simulation context |
| **Can Trigger Resampling** | Yes (validation failures) | No |
| **Performance Critical** | Less (runs ~100-1000×) | More (could run 100,000×+) |
| **Purpose** | Constrain/transform sampled values | Prepare for simulator interface |
| **Signature** | `apply(theta: dict) -> dict` | `apply(theta, model_config, n_trials) -> dict` |
| **Typical Operations** | Math constraints, scaling, normalization | Dimension expansion, renaming, defaults |

---

## Arguments FOR Unification

### 1. Conceptual Simplicity
- Users only learn one system for parameter manipulation
- Less cognitive overhead

### 2. Code Maintenance
- One codebase instead of two
- Fewer tests to maintain
- Single documentation source

### 3. Avoid Confusion
- Current naming is confusing ("transforms" vs "processors")
- Some operations could fit in either category

---

## Arguments AGAINST Unification (Why Separation is Justified)

### 1. **Fundamental Phase Separation** ⭐⭐⭐
**Most important argument**

The two systems operate at fundamentally different phases of the pipeline:

```
Parameter Sampling → [TRANSFORMS] → Store θ → ... → Load θ → [PROCESSORS] → Simulate
     ^                                                              ^
     |                                                              |
Once per θ set                                              Many times per θ set
```

Moving transforms to simulation phase would:
- Run constraint checks repeatedly (performance hit)
- Lose the ability to resample invalid parameter combinations
- Mix data generation logic with simulation logic

Moving processors to sampling phase would:
- Lose access to `n_trials` (simulation-specific)
- Force dimension expansion before knowing simulation context
- Break the clean separation between sampling and execution

### 2. **Different Performance Requirements** ⭐⭐⭐

**Parameter Transforms**: Run ~100-1000 times during data generation
- Can afford more expensive operations
- Constraint checking, validation acceptable here

**Theta Processors**: Run potentially 100,000+ times
- Must be very fast
- No validation, just preparation
- Example: Generating 100 parameter sets × 10 subdatasets × 100 batch calls = 100,000 simulator calls

### 3. **Different Semantic Purposes** ⭐⭐
**Parameter Transforms**: Enforce mathematical/statistical constraints
- "This parameter combination is invalid, resample"
- "Transform from sampling space to parameter space"
- About the **validity** of parameters

**Theta Processors**: Prepare for simulator interface
- "This simulator expects 2D arrays, expand dimensions"
- "This simulator uses name 'z' not 'A'"
- About the **format** of parameters

### 4. **Validation vs Preparation** ⭐⭐

**Transforms** can raise errors → trigger resampling
```python
if theta['a'] < theta['z']:
    raise ParameterValidationError("Invalid: a < z")  # Resample!
```

**Processors** cannot trigger resampling - simulation is already committed
```python
# Just prepare - can't go back to sampling phase
theta['v'] = np.expand_dims(theta['v'], axis=1)
```

### 5. **Different Input/Output Contracts** ⭐

**Transforms**:
```python
def apply(theta: dict) -> dict:
    # No context needed
    return modified_theta
```

**Processors**:
```python
def apply(theta: dict, model_config: dict, n_trials: int) -> dict:
    # Needs simulation context
    return prepared_theta
```

Unifying would require:
- Either giving transforms unnecessary context
- Or denying processors necessary context

### 6. **Separation of Concerns** ⭐

Clean architectural boundaries:
- **Parameter Samplers** → Sample + Transform → Valid θ
- **Simulators** → Process → Execute → Results

Mixing them would blur:
- What happens during data generation?
- What happens during simulation?
- Who is responsible for what?

---

## Real-World Examples

### Why Transforms Can't Be Moved to Simulation

**Example 1: Constraint Enforcement**
```python
# Parameter transform: Ensure a > z
# If violated, RESAMPLE different (a, z) pair
transform = SwapIfLessTransform("a", "z")
```

If this ran during simulation:
- We've already committed to these (a, z) values
- Can't go back to sampling
- Would need to either: (1) simulate anyway (invalid!), or (2) return error to data generator (messy!)

**Example 2: Log-Space Sampling**
```python
# Sample v from [-3, 3] (log-space for symmetric sampling)
# Transform to real space: v = exp(v)
```

This MUST happen during sampling, not simulation, because:
- It's about the sampling strategy (symmetric in log-space)
- Simulator expects real values, not log values
- Part of the parameter generation logic

### Why Processors Can't Be Moved to Sampling

**Example 1: Trial Expansion**
```python
# Expand dimensions to match n_trials
theta['v'] = np.tile(theta['v'], (n_trials, 1))
```

Can't do this during sampling because:
- We don't know `n_trials` yet (simulation-specific)
- Different simulations might use different `n_trials`
- Same parameter set might be simulated with various `n_trials`

**Example 2: Model-Specific Preparation**
```python
# LBA model needs v0, v1 stacked into 2D v array
theta['v'] = np.column_stack([theta['v0'], theta['v1']])
```

Can't do this during sampling because:
- Sampler doesn't know target model's interface
- Same θ might be used with different simulators
- Simulator interface is execution concern, not sampling concern

---

## Areas Where Boundary is Unclear

### 1. SetDefaultValue
Current: Theta Processor
Could arguably be: Parameter Transform (set during sampling)

**Analysis**: Currently in processor because defaults might depend on `n_trials` (e.g., `t = np.zeros(n_trials)`)

### 2. Parameter Renaming
Current: Theta Processor
Could arguably be: Parameter Transform

**Analysis**: In processor because renaming is about simulator interface, not parameter validity

### 3. Simple Scaling/Clipping
Current: Could be either
Example: `theta['v'] = np.clip(theta['v'], -5, 5)`

**Analysis**:
- If about **valid parameter space** → Transform
- If about **simulator numerical stability** → Processor

---

## Alternative Unification Approaches (All Problematic)

### Approach 1: Everything in Sampling
```python
# Apply all transforms and processors during sampling
param_sampler.sample(n_samples=1, n_trials=1000)  # Need to know n_trials during sampling!
```

**Problems**:
- Couples sampling to simulation details
- Can't reuse same θ with different `n_trials`
- Forces early binding of simulation parameters

### Approach 2: Everything in Simulation
```python
# Move constraint enforcement to simulation
def simulate(theta, ...):
    if theta['a'] < theta['z']:
        raise ValueError("Invalid parameters")  # Can't resample here!
```

**Problems**:
- Can't trigger resampling from inside simulator
- Violates single responsibility (simulator shouldn't validate sampling)
- Would need to bubble errors up to data generator (complex)

### Approach 3: Unified Transformation System with Phases
```python
class UnifiedTransform:
    phase: str  # "sampling" or "simulation"

    def apply(self, theta, context=None):
        if self.phase == "sampling":
            # Run during sampling
        else:
            # Run during simulation (need context)
```

**Problems**:
- More complex than current system
- Users need to understand phases anyway
- Doesn't actually eliminate the conceptual separation
- Just hides it behind a flag

---

## Counter-Arguments Addressed

### "Users find it confusing to have both"

**Response**: Rename for clarity rather than merge:
- **Parameter Transforms** → **Sampling Constraints** or **Post-Sample Transforms**
- **Theta Processors** → **Simulation Preparers** or **Pre-Simulation Transforms**

The confusion is in naming, not in the architectural separation.

### "Some operations could fit in either"

**Response**: True, but:
1. Most operations have a clear home (90%+ of cases)
2. For ambiguous cases, use this decision tree:
   - Does it validate/constrain sampled values? → Transform
   - Does it prepare for simulator interface? → Processor
   - Does it need `n_trials` or `model_config`? → Processor
   - Could it trigger resampling? → Transform

### "It's more code to maintain"

**Response**: The code performs fundamentally different functions. Merging would:
- Not reduce total code volume significantly
- Make each piece more complex (conditional logic for phases)
- Reduce clarity about what runs when

---

## Recommendations

### 1. **Keep Separation** ✅
Maintain architectural distinction between sampling and simulation phases.

### 2. **Improve Naming**
Consider renaming for clarity:
- `ParameterTransform` → `SamplingConstraint` or `PostSampleTransform`
- `ThetaProcessor` → `SimulationPreparer` or `PreSimulationTransform`

### 3. **Better Documentation**
Create clear guide explaining:
- When to use each
- Decision tree for ambiguous cases
- Why both exist

### 4. **Potential Optimization**
For performance-critical paths, cache processor results when:
- Same `θ` + `model_config` + `n_trials` combination
- Processors are deterministic
- Could save repeated work

### 5. **Clear Guidelines**
Document decision rules for new transforms:

**Use Parameter Transform if**:
- ✅ Validates parameter relationships
- ✅ Transforms from sampling space to parameter space
- ✅ Could trigger resampling
- ✅ Doesn't need `n_trials` or simulation context

**Use Theta Processor if**:
- ✅ Prepares parameters for simulator interface
- ✅ Handles dimension/shape requirements
- ✅ Sets defaults based on `n_trials`
- ✅ Model-specific parameter renaming/stacking

---

## Conclusion

While parameter transforms and theta processors may appear redundant, they reflect a fundamental architectural separation between **data generation** and **simulation execution**. This separation:

1. **Improves performance** (expensive operations run once, not repeatedly)
2. **Enables validation** (can resample invalid parameters)
3. **Maintains clear boundaries** (sampling logic ≠ simulation logic)
4. **Provides flexibility** (same θ can be simulated with different contexts)

The apparent redundancy is actually **separation of concerns** - a software engineering best practice. Rather than unifying, we should:
- **Clarify naming** to reduce confusion
- **Document boundaries** more explicitly
- **Provide clear guidelines** for when to use each

**Verdict**: The dual system is justified and should be maintained.

---

## Appendix: Performance Analysis

### Cost of Running Processors Repeatedly

**Scenario**: Generate training data for 100 parameter sets

```
Without transforms (all in processors):
- 100 parameter sets
- 10 subdatasets per set
- 100 simulator calls per subdataset
= 100 × 10 × 100 = 100,000 processor executions

With transforms (separated):
- 100 parameter transforms (during sampling)
- 100 × 10 × 100 = 100,000 processor executions (but simpler/faster)
```

**Key insight**: Moving constraint checking from transforms (100×) to processors (100,000×) would be a 1000× performance hit for validation logic.

### Memory Efficiency

**Current**: Sample θ → Transform once → Store → Reuse many times
**Unified**: Would need to either:
1. Store expanded/processed θ (much larger memory footprint)
2. Re-process every time (performance hit)
