from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

from .schema import BaseConfigSchema, HSSMConfigSchema, RLSSMConfigSchema

C = TypeVar("C", bound=BaseConfigSchema)

HSSM_INTERNAL_MODELS = [
    "ddm",
    "ddm_sdv",
    "full_ddm",
    "angle",
    "levy",
    "ornstein",
    "weibull",
    "race_no_bias_angle_4",
    "ddm_seq2_no_bias",
    "gamma_drift",
    "lba3",
    "lba4",
    "lba2",
    "racing_diffusion_3",
    "poisson_race",
    "softmax_inv_temperature_2",
    "softmax_inv_temperature_3",
]

# TODO: Add RLSSM supported models to the registry once they are implemented.
RLSSM_INTERNAL_MODELS = [
    "rlddm",
    "rlwm",
]


class BaseModelRegistry(Generic[C]):
    """
    A registry for storing and retrieving SSM configuration schemas.
    """

    initialized: bool = False
    prefix: ClassVar[str]
    base_path: ClassVar[Path]
    model_schema: type[C]
    internal_models: ClassVar[list[str]]
    external_models: ClassVar[dict[str, str | Path]]
    _instance: ClassVar["BaseModelRegistry[Any] | None"] = None

    def __new__(cls) -> Self:
        # One instance per concrete registry class. cls.__dict__ rather than
        # cls._instance: plain attribute lookup walks the MRO and would hand a
        # subclass whatever instance the base class happens to hold.
        instance = cls.__dict__.get("_instance")
        if instance is None:
            instance = super().__new__(cls)
            cls._instance = instance
        return instance

    def __init__(self):
        # __new__ returns the same object every time, so this guard makes the
        # (re-)initialization a no-op after the first construction.
        if self.initialized:
            return
        self.initialized = True

    def __init_subclass__(cls, prefix: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.prefix = prefix
        cls.base_path = Path(__file__).parent / prefix
        # Fresh per subclass: a dict on the base class would be shared by every
        # registry, so an external HSSM model would leak into the RLSSM one.
        cls.external_models = {}
        cls._instance = None

    def is_internal(self, name: str) -> bool:
        """Check if a model is supported by the registry.

        Parameters
        ----------
        name : str
            The name of the model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        return name in self.internal_models

    def is_external(self, name: str) -> bool:
        """Check if a model is external to the registry.

        Parameters
        ----------
        name : str
            The name of the model to check.

        Returns
        -------
        bool
            True if the model is external, False otherwise.
        """
        return name in self.external_models

    def is_supported(self, name: str) -> bool:
        """Check if a model is supported by the registry.

        Parameters
        ----------
        name : str
            The name of the model to check.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        return self.is_internal(name) or self.is_external(name)

    def load_config(self, name: str) -> C:
        """Load a JSON configuration file from disk.

        Parameters
        ----------
        name : str
            The name of the model to load.

        Returns
        -------
        C
            The loaded configuration schema object.
        """
        if self.is_internal(name):
            file_path = self.base_path / f"{name}.json"
        elif self.is_external(name):
            file_path = Path(self.external_models[name])
        else:
            raise ValueError(f"Model '{name}' is not supported by the registry.")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Configuration file for model '{name}' not found at {file_path}"
            )

        # Fast JSON parsing and validation using Pydantic's model_validate_json method
        return self.model_schema.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )


class HSSMRegistry(BaseModelRegistry[HSSMConfigSchema], prefix="hssm"):
    """
    A registry for storing and retrieving HSSM configuration schemas.
    """

    model_schema = HSSMConfigSchema
    internal_models = HSSM_INTERNAL_MODELS


class RLSSMRegistry(BaseModelRegistry[RLSSMConfigSchema], prefix="rlssm"):
    """
    A registry for storing and retrieving RLSSM configuration schemas.
    """

    model_schema = RLSSMConfigSchema
    internal_models = RLSSM_INTERNAL_MODELS


hssm_registry = HSSMRegistry()
rlssm_registry = RLSSMRegistry()
