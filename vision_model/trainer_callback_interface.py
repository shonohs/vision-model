import abc
import typing
import torch


class TrainerCallbackInterface(abc.ABC):
    """Interfaces a Trainer provides for callback methods"""
    @property
    @abc.abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get a training dataloader."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def val_dataloader(self) -> typing.Optional[torch.utils.data.DataLoader]:
        """Get a validation dataloader.

        Returns None if a validation set is not available.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError
