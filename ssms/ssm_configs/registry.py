from pathlib import Path
from typing import TypeVar

from .schema import BaseConfigSchema

ConfigType = TypeVar("ConfigType", bound=BaseConfigSchema)

class BaseModelRegistry:
    """
    A registry for storing and retrieving SSM configuration schemas.
    """

    initialized: bool = False
    model_schema: type[BaseConfigSchema]
    internal_models: list[str]
    external_models = {}

    def __init__(self):
        # Ensures that the registry is a singleton and only initialized once.
        if self.initialized:
            return
        self.initialized = True

    def __init__subclass__(cls, prefix: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.prefix = prefix
        cls.base_path = Path(__file__).parent / prefix

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

    def load_config(self, name: str) -> BaseConfigSchema:
        """Load a JSON configuration file from disk.

        Parameters
        ----------
        name : str
            The name of the model to load.

        Returns
        -------
        dict
            The loaded JSON configuration as Pydantic model instance.
        """
        if self.is_internal(name):
            file_path = self.base_path / f"{name}.json"
        elif self.is_external(name):
            file_path = Path(self.external_models[name])
        else:
            raise ValueError(f"Model '{name}' is not supported by the registry.")

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file for model '{name}' not found at {file_path}")

        # Fast JSON parsing and validation using Pydantic's model_validate_json method
        return self.model_schema.model_validate_json(file_path.read_text(encoding="utf-8"))
