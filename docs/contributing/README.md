# Contributing to SSM-Simulators

Thank you for your interest in contributing to `ssm-simulators`! This directory contains guides to help you contribute effectively.

## Contribution Guides

### Adding New Models

The main contribution pathway for researchers is adding new sequential sampling models to the package.

**[📖 Complete Guide: Adding New Models](add_models.md)**

This comprehensive tutorial covers:
- **Level 1**: Boundary/Drift Variants - Add model variants using existing simulators (~15 min)
- **Level 2**: Python Simulators - Implement new models in Python (~20 min)
- **Level 3**: Cython Simulators - Create high-performance implementations (~30 min)

Each level includes:
- Step-by-step instructions with code examples
- Testing requirements and templates
- Documentation guidelines
- PR submission checklist

### Custom Parameter Adaptations

**[📖 Guide: Custom Parameter Adaptations](add_parameter_adapters.md)**

Learn how to create custom parameter transformations for your models. This guide covers:
- Using built-in parameter adaptations
- Creating custom adaptations
- Real-world examples and best practices

### Other Ways to Contribute

- **Bug Reports**: Found a bug? [Open an issue](https://github.com/lnccbrown/ssm-simulators/issues) with a minimal reproducible example
- **Documentation**: Improve tutorials, fix typos, add examples
- **Feature Requests**: Suggest new features or improvements
- **Code Review**: Review open pull requests

## Quick Links

- [Main README](../index.md)
- [API Documentation](../api/)
- [Tutorials](../core_tutorials/)
- [GitHub Issues](https://github.com/lnccbrown/ssm-simulators/issues)
- [Pull Requests](https://github.com/lnccbrown/ssm-simulators/pulls)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (requires [uv](https://github.com/astral-sh/uv)):
   ```bash
   cd ssm-simulators
   uv pip install -e ".[dev]"
   ```
   Or, if you want to sync *all* dependency groups for development:
   ```bash
   uv sync --all-groups
   ```
   If you don't have `uv` yet:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. **Create a branch** for your contribution
5. **Make your changes** following the relevant guide
6. **Run tests** to ensure nothing broke
7. **Submit a pull request** with a clear description

## Questions?

If you have questions about contributing:
- Search [existing issues](https://github.com/lnccbrown/ssm-simulators/issues)
- Post in our [GitHub Discussions forum](https://github.com/lnccbrown/ssm-simulators/discussions) for general questions or help
- Open a new issue with the `question` label if needed

Thank you for helping make `ssm-simulators` better for the research community!
