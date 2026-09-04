# hssm-example-model

A minimal, working example of an `ssm-configs` plugin. Copy this directory as
the starting point for your own model package.

There are only three moving parts:

1. **The distribution name** — `hssm-example-model`. The `hssm-` prefix selects
   the registry (`ssm_configs.hssm_registry`); the rest becomes the model name,
   `example_model`. One distribution ships one model, and the name is not
   repeated anywhere else.
2. **The entry point** — `[project.entry-points."ssm_configs"]` in
   `pyproject.toml`, pointing at the module that implements the hook. This is
   what lets `ssm_configs` find the package without anyone importing it.
3. **The hook** — `ssm_configs_config_path()`, returning the path to the JSON
   config shipped inside the package.

## Trying it

```bash
uv sync --all-groups          # installs this package into the workspace env
uv run python -c "
import ssm_configs
print(ssm_configs.hssm_registry.is_external('example_model'))
print(ssm_configs.hssm_registry.load_config('example_model'))
"
```

Note that the snippet never imports `hssm_example_model`. Installation alone is
enough.

## Making it your own

Rename the distribution in `pyproject.toml` (`hssm-my-cool-model`), rename the
module directory to match (`hssm_my_cool_model`), point the entry point at it,
and replace `example_model.json` with your own config. The model name follows
from the distribution name automatically.

If your model belongs in a different registry, lead the distribution name with
that registry's prefix instead — `rlssm-my-cool-model` lands in
`ssm_configs.rlssm_registry`.
