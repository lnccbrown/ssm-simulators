"""RLSSM simulation framework for ssm-simulators.

Provides modular, compositional simulation of Reinforcement Learning -
Sequential Sampling Models (RLSSM).
"""

from ._registry import get_rlssm_preset, list_rlssm_presets, register_rlssm_preset
from .learning_process import LearningProcess, RescorlaWagnerDeltaRule
from .rl_config import RLSSMModelConfig
from .rl_simulator import RLSSMSimulator
from .task_environment import TaskConfig, TaskEnvironment, TwoArmedBandit

__all__ = [
    # Core classes
    "RLSSMSimulator",
    "RLSSMModelConfig",
    # Protocols
    "LearningProcess",
    "TaskEnvironment",
    # Built-in implementations
    "RescorlaWagnerDeltaRule",
    "TwoArmedBandit",
    "TaskConfig",
    # Registry
    "get_rlssm_preset",
    "list_rlssm_presets",
    "register_rlssm_preset",
]
