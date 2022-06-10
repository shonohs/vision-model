from typing import Dict, List, Union
import torch
from .plugin_base import PluginBase


class ModelBase(torch.nn.Module):
    """Base class for vision-related models.

    Attributes:
        HYPERPARAMETERS (Dict): Model specific hyper parameters.
            A model can provide names of model-specific hyperparameters and their candidates.
            If provided, trainer may run multiple trainings searching for the best hyperparameters.

            Example:
            HYPERPARAMETERS = {'dropout_rate': [0.1, 0.5, 0.9]}
    """
    HYPERPARAMETERS: Dict[str, Union[List[int], List[float], List[str]]] = {}

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
