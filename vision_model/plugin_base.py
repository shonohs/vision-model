from typing import Any, Tuple
import typing
from .trainer_callback_interface import TrainerCallbackInterface


if typing.TYPE_CHECKING:
    from .model_base import ModelBase


class PluginBase:
    """Base class for plugins to modify Trainer behavior."""

    def on_train_start(self, trainer: TrainerCallbackInterface, model: 'ModelBase') -> None:
        """Called when the training begins.

        Optimizers are configured after this callback is called.
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
