from dataclasses import field

from pydantic import BaseModel

# ====== Centralized SSM defaults =====
DEFAULT_SSM_OBSERVED_DATA = ["rt", "response"]
DEFAULT_SSM_CHOICES = (0, 1)


# An example of a base config schema for SSMs. This can be extended for specific SSMs such as RLSSMs.
class BaseConfigSchema(BaseModel):
    name: str
    description: str | None = None

    response: list[str] = DEFAULT_SSM_OBSERVED_DATA
    choices: tuple[int, ...] | None = DEFAULT_SSM_CHOICES

    list_params: dict[str, list] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # TODO: Add any additional common parameters for all models here


class HSSMConfigSchema(BaseConfigSchema):
    """Configuration schema for HSSM models."""

    # TODO: Add any HSSM-specific parameters here
    pass


class RLSSMConfigSchema(BaseConfigSchema):
    """Configuration schema for RLSSM models."""

    # TODO: Add any RLSSM-specific parameters here
    pass
