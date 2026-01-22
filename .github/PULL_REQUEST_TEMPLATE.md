## New Model: [Your Model Name]

<!--
Thank you for contributing to ssm-simulators!
Please fill out this template to help us review your contribution.
For guidance, see: docs/contributing/add_models.md

Note: This template is primarily for PRs that concern the addition of new models.
If your contribution is not of that flavor, feel free to ignore the specifics of the
template below, but leave enough context for a reviewer to understand what the PR
is about.
-->

### Description

<!-- 1-2 paragraph description of the model and its purpose -->

### Type of Contribution

<!-- Check the box that applies -->

- [ ] Level 1: Boundary/Drift variant
- [ ] Level 2: Python simulator
- [ ] Level 3: Cython simulator

### Model Details

- **Parameters**: <!-- list parameters with brief descriptions, e.g., v (drift rate), a (boundary) -->
- **Number of choices**: <!-- e.g., 2 for binary decision -->
- **Reference**: <!-- paper citation(s), e.g., Smith et al. (2024). Journal, DOI/link -->
- **Use case**: <!-- when to use this model vs alternatives -->

### Correctness Validation

<!-- Check all that apply -->

- [ ] Tested against theoretical predictions (mean, variance, etc.)
- [ ] Compared with published results (if available)
- [ ] Edge cases tested and handled
- [ ] Statistical properties validated with large samples

**Validation details**: <!-- Briefly explain how you validated correctness -->


### Testing

- **Tests written**: <!-- Yes/No -->
- **All tests pass**: <!-- Yes/No -->
- **Test coverage**: <!-- X% if known, or "not measured" -->
- **Performance benchmark** (if Cython): <!-- X samples/sec, or "N/A" for Python -->

### Documentation

- [ ] Model config has comprehensive docstring
- [ ] References to papers/equations included
- [ ] Parameter meanings and ranges documented
- [ ] Example usage provided
- [ ] Tutorial notebook: <!-- Yes/No/Planned -->

### Pre-Submission Checklist

<!-- Verify these before submitting -->

- [ ] Code follows existing style and patterns
- [ ] All new tests pass locally
- [ ] Existing tests still pass (`pytest tests/`)
- [ ] Model is registered and importable
- [ ] No unnecessary files committed (temp files, notebook outputs, etc.)
- [ ] Commit messages are clear and descriptive

### Additional Notes

<!-- Any other information relevant to reviewers -->


---

<!--
Review Guidelines for Maintainers:
1. Verify mathematical correctness against cited references
2. Check test coverage includes edge cases
3. Validate documentation is complete and clear
4. Ensure code follows project patterns
5. Test on multiple platforms if Cython
-->
