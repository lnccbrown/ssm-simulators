# PyDDM Auxiliary Labels Implementation

## Overview

Extended the `PyDDMGenerationStrategy` to compute auxiliary training labels analytically, matching the output format of `SimulationBasedGenerationStrategy` while maintaining the efficiency benefits of analytical solutions.

## Motivation

Previously, `PyDDMGenerationStrategy` only generated LAN (likelihood approximation network) data, setting all auxiliary labels (CPN, OPN, GoNoGo) to `None`. This created an inconsistency with `SimulationBasedGenerationStrategy` and limited the usefulness of PyDDM-based data generation for multi-task learning scenarios.

However, PyDDM's analytical solutions can compute many of these auxiliary labels **without running trajectory simulations**, by numerically integrating the probability density functions.

## Implementation

### Key Changes

1. **Added `_compute_auxiliary_labels()` method** to `PyDDMGenerationStrategy`
   - Computes choice probabilities analytically using `solution.prob("correct")` and `solution.prob("error")`
   - Handles deadline logic by integrating PDFs pre/post deadline
   - Computes omission probabilities and go/nogo probabilities
   - Based on the implementation in `notebooks/pyddm_tests.ipynb` (Cell 7)

2. **Updated `generate_for_parameter_set()` method**
   - Calls `_compute_auxiliary_labels()` after building the estimator
   - Populates all auxiliary label fields in the result dictionary
   - Maintains `binned_128` and `binned_256` as `None` (these require trajectory data)

3. **Updated class docstring**
   - Clarifies that auxiliary labels are computed analytically
   - Notes that only binned histograms are unavailable

### Auxiliary Labels Computed

| Label | Description | Computation Method |
|-------|-------------|-------------------|
| `cpn_labels` | Choice probabilities [error, correct] | `solution.prob()` or integrate PDFs pre-deadline |
| `cpn_no_omission_labels` | Choice probabilities excluding omissions | Renormalize pre-deadline probabilities |
| `opn_labels` | Omission probability (RT beyond deadline) | Integrate PDFs post-deadline |
| `gonogo_labels` | No-go probability (error + post-deadline correct) | `P(error) + P(correct, post-deadline)` |

### Output Format Consistency

Both strategies now produce identical output keys:

```python
{
    "lan_data": np.ndarray,          # Training RT/choice pairs
    "lan_labels": np.ndarray,        # Log-likelihoods
    "cpn_data": np.ndarray,          # Parameters (theta)
    "cpn_labels": np.ndarray,        # Choice probabilities (1, 2)
    "cpn_no_omission_data": np.ndarray,
    "cpn_no_omission_labels": np.ndarray,  # (1, 2)
    "opn_data": np.ndarray,          # Parameters (theta)
    "opn_labels": np.ndarray,        # Omission prob (1, 1)
    "gonogo_data": np.ndarray,       # Parameters (theta)
    "gonogo_labels": np.ndarray,     # Nogo prob (1, 1)
    "binned_128": None or np.ndarray,  # None for PyDDM, array for simulation
    "binned_256": None or np.ndarray,  # None for PyDDM, array for simulation
    "theta": np.ndarray,             # Parameter array
}
```

## Deadline Handling

The implementation correctly handles two cases:

### Case 1: No Deadline (or deadline ≥ max_t)

```python
choice_p[0, 0] = solution.prob("error")
choice_p[0, 1] = solution.prob("correct")
choice_p_no_omission = choice_p  # Same as choice_p
nogo_p = choice_p[0, 0]          # Just P(error)
omission_p = 0.0                  # No omissions
```

### Case 2: Deadline < max_t

```python
# Split probabilities at deadline
choice_p_pre = integrate.simpson(pdf[t < deadline], t[t < deadline])
choice_p_post = solution.prob(choice) - choice_p_pre

# Overall choice probabilities (pre-deadline only)
choice_p = choice_p_pre

# Renormalize excluding omissions
choice_p_no_omission = choice_p_pre / sum(choice_p_pre)

# Omission = any response beyond deadline
omission_p = choice_p_post[error] + choice_p_post[correct]

# Nogo = error boundary OR post-deadline correct
nogo_p = choice_p_post[correct] + choice_p[error]
```

## Benefits

1. **Consistency**: Both strategies now produce the same output structure
2. **Efficiency**: PyDDM computes auxiliary labels analytically (no simulation overhead)
3. **Accuracy**: Uses exact analytical solutions where available
4. **Completeness**: Enables multi-task learning with PyDDM data
5. **Maintainability**: Single code path for processing outputs downstream

## Testing

Comprehensive test suite added in `tests/dataset_generators/test_pyddm_auxiliary_labels.py`:

- ✓ Auxiliary labels exist and are not None (except binned)
- ✓ Correct shapes: `(1, 2)` for choice probs, `(1, 1)` for omission/nogo
- ✓ Choice probabilities sum to 1
- ✓ All probabilities in [0, 1]
- ✓ `nogo_p = P(error)` when no deadline
- ✓ `omission_p = 0` when no deadline
- ✓ Output format matches `SimulationBasedGenerationStrategy`

## Example Usage

```python
from ssms.dataset_generators import DataGenerator
from ssms.config import model_config, get_lan_config

# Configure for PyDDM
config = get_lan_config()
config['model'] = 'ddm'
config['estimator_type'] = 'pyddm'

# Create data generator
dg = DataGenerator(
    generator_config=config,
    model_config=model_config['ddm']
)

# Generate data with full auxiliary labels
data = dg.generate_data_training_uniform(n_samples=1000, n_parameter_sets=100)

# Access auxiliary labels
choice_probs = data['cpn_labels']      # Shape: (100, 2)
omission_probs = data['opn_labels']    # Shape: (100, 1)
nogo_probs = data['gonogo_labels']     # Shape: (100, 1)
```

## Files Modified

- `ssms/dataset_generators/strategies/pyddm_strategy.py`
  - Added `_compute_auxiliary_labels()` method
  - Updated `generate_for_parameter_set()` to call it
  - Updated class docstring

## Files Added

- `tests/dataset_generators/test_pyddm_auxiliary_labels.py`
  - Comprehensive test suite for auxiliary label computation

## Limitations

- **Binned RT histograms** (`binned_128`, `binned_256`) remain `None` for PyDDM
  - These require full RT trajectory distributions separated by choice
  - Would need to sample from PyDDM solution and bin, losing the analytical efficiency
  - Current use case (as noted by user) doesn't require these outputs

## Future Enhancements

If binned histograms are needed in the future, they could be computed by:
1. Sampling N trajectories from `solution.resample()`
2. Binning RTs by choice
3. Trading some analytical efficiency for completeness

However, this is not currently required for downstream workflows.

## References

- Implementation based on `notebooks/pyddm_tests.ipynb` Cell 7
- Consistent with `SimulationBasedGenerationStrategy._compute_auxiliary_labels()`
- PyDDM documentation: https://pyddm.readthedocs.io/
