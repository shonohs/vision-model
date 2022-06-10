import abc
import typing
import torch


class TrainerCallbackInterface(abc.ABC):
    """Interfaces a Trainer provides for callback methods"""
    @property
    @abc.abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def val_dataloader(self) -> typing.Optional[torch.utils.data.DataLoader]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_epochs(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def criterion(self) -> torch.nn.Module:
        raise NotImplementedError

    @criterion.setter
    @abc.abstractmethod
    def criterion(self, value):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def predictor(self) -> torch.nn.Module:
        raise NotImplementedError

    @predictor.setter
    @abc.abstractmethod
    def predictor(self, value):
        raise NotImplementedError
