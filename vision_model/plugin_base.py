from typing import Any, Dict, Tuple
import typing
import torch
from .trainer_callback_interface import TrainerCallbackInterface


if typing.TYPE_CHECKING:
    from .model_base import ModelBase


class PluginBase:
    """Callbacks to modify the training behavior."""

    def on_train_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the training begins.
           Notes:
               - Optimizers are configured after this callback is called.
        """
        pass

    def on_train_end(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the training ends."""
        pass

    def on_train_batch_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase', batch: Tuple, batch_index: int) -> Any:
        """Called when the train batch begins. Returns a modified batch."""
        return batch

    def on_train_batch_end(self, trainer: TrainerCallbackInterface, model: 'ModelBase', batch: Tuple, batch_index: int) -> None:
        """Called when the train batch ends."""
        pass

    def on_train_epoch_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the train epoch ends."""
        pass

    def on_prediction_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the prediction begins."""
        pass

    def on_prediction_end(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the prediction ends."""
        pass

    def on_train_backward_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase', loss: torch.Tensor) -> torch.Tensor:
        """Called when the train backward step begins. Returns a modified loss."""
        return loss

    def on_save_onnx_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase', nn_model: torch.nn.Module) -> torch.nn.Module:
        """Called before ONNX tracing is performed. The returned nn_model will be used for tracing."""
        return nn_model

    def on_save_onnx_end(self, trainer: TrainerCallbackInterface, model: 'ModelBase', onnx_model: bytes) -> bytes:
        """Called when the model is saved as onnx. Returns an updated onnx model binary."""
        return onnx_model

    def on_save_model(self, trainer: TrainerCallbackInterface, model: 'ModelBase', state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Called when the trainer saves a pytorch model state."""
        return state_dict
