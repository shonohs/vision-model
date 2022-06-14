from typing import Dict, List, Union
import torch
from .plugin_base import PluginBase


class ModelBase(torch.nn.Module):
    """Base class for vision-related models that support both training and inference."""
    @staticmethod
    def get_hyperparameter_spaces() -> Dict[str, Union[List[int], List[float], List[str]]]:
        """Get model-specific hyperparamter spaces.

        A model can provide names of model-specific hyperparameters and their candidates.
        If provided, trainer may run multiple trainings searching for the best hyperparameters.

        Examples:
            {'dropout_rate': [0.1, 0.5, 0.9]}
        """

        return {}

    @property
    def plugins(self) -> List[PluginBase]:
        """List of plugins that are required for the model"""
        return []

    @property
    def criterion(self) -> torch.nn.Module:
        """Get a criterion module for the model. Input/Output formats are defined for each task."""
        raise NotImplementedError

    @property
    def predictor(self) -> torch.nn.Module:
        """Get a predictor module for the model. Input/Output formats are defined for each task."""
        raise NotImplementedError


class InferenceModelBase(torch.nn.Module):
    """Base class for inference-only models."""
    @property
    def predictor(self) -> torch.nn.Module:
        """Get a predictor module for the model. Input/Output formats are defined for each task."""
        raise NotImplementedError
