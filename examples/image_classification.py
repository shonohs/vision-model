import torch
import torchvision.models
from vision_model import ModelBase


class MobileNetV2(ModelBase):
    def __init__(self, num_classes):
        self._model = torchvision.models.mobilenet_v2(num_classes=num_classes)
        self._mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        self._std = torch.Tensor([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    def forward(self, x):
        x = (x - self._mean) / self._std
        return self._model(x)

    def load_state_dict(self, *args, **kwargs):
        self._model.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self._model.state_dict()

    @property
    def criterion(self):
        return torch.nn.CrossEntropyLoss()

    @property
    def predictor(self):
        return torch.nn.Softmax(1)
