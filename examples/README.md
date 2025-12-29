# Examples

This directory contains example scripts demonstrating various features of the `ssm-simulators` package.

## Available Examples

### `custom_transforms_example.py`

Comprehensive demonstration of the custom parameter transform system, including:

- **Function-based transforms**: Simple transformations without configuration
- **Class-based transforms**: Configurable transformations with parameters
- **Transform factories**: Programmatically creating multiple similar transforms
- **Multi-parameter transforms**: Enforcing constraints between parameters
- **Integration with DataGenerator**: Complete workflow from registration to data generation

**Run the example:**

```bash
cd examples
python custom_transforms_example.py
```

**Key takeaways:**
- Register transforms before creating `DataGenerator`
- Use descriptive names for clarity
- Mix custom and built-in transforms freely
- Transforms work automatically with array inputs

## Quick Start

To use custom transforms in your own code:

```python
from ssms import register_transform_function
import numpy as np

# 1. Register your transform
def my_transform(theta: dict) -> dict:
    if 'v' in theta:
        theta['v'] = np.exp(theta['v'])
    return theta

register_transform_function("my_transform", my_transform)

# 2. Use it in a model config
model_config = {
    "name": "my_model",
    "params": ["v", "a", "z", "t"],
    "param_bounds": [...],
    "parameter_transforms": [
        {"type": "my_transform"}
    ]
}

# 3. Generate data as usual
from ssms.dataset_generators.lan_mlp import DataGenerator
generator = DataGenerator(model_config=model_config, ...)
data = generator.generate_data_training_uniform(n_training_samples=1000)
```

## Documentation

For detailed documentation, see:
- [Custom Transforms Guide](../docs/custom_transforms.md) - Comprehensive guide with best practices
- [Parameter Sampler Refactor Summary](../PARAMETER_SAMPLER_REFACTOR_SUMMARY.md) - Architecture overview
- [API Documentation](../docs/api/dataset_generators.md) - Full API reference

## Contributing Examples

If you have useful examples to share, please consider contributing them! Examples should:
- Be well-commented and explain the "why" not just the "what"
- Run without errors on a fresh install
- Demonstrate practical, real-world use cases
- Follow PEP 8 style guidelines
