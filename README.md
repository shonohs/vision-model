# Common interfaces for vision models.

This package provides common interfaces for vision models. 

Each Model must have two additional torch.nn.Module; criterion and predictor. A criterion is used in the training phase to calculate the loss from the model outputs. A predictor is used in test phase to get final prediction results from the model outputs. A Model must define criterion() and predictor() property to return a corresponding module. The expected input/output formats for model, criterion, predictor are defined for each task.

When a model requires advanced changes to training behavior, it can implement a Plugin class inheriting PluginBase. For the detail, please see the description of the PluginBase class.

## Install
```
pip install vision-model
```

## Examples

Here is an example to define a simple multiclass classification model.
```python
from vision_model import ModelBase, PluginBase

class MyModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequence([torch.nn.Conv2d(...)])
    
    def forward(self, x):
        return self.model(x)

    def change_something(self):
        pass  # Do something here if needed.

    @property
    def plugins(self):
        return [MyPlugin()]

    @property
    def criterion(self):
        return torch.nn.CrossEntropyLoss()

    @property
    def predictor(self):
        return torch.nn.Softmax(1)


class MyPlugin(PluginBase):
    def on_train_start(self, trainer, model):
        # It can make some changes to the model when a training begins.
        # See help(PluginBase) to find available callback methods.
        model.change_something()
        return
```

Pleasee see examples/ directory for more practical examples.


## Interfaces

```
class ModelBase(torch.nn.modules.module.Module)
 |  Base class for vision-related models.
 |
 |  Attributes:
 |      HYPERPARAMETERS (Dict): Model specific hyper parameters.
 |          A model can provide names of model-specific hyperparameters and their candidates.
 |          If provided, trainer may run multiple trainings searching for the best hyperparameters.
 |
 |          Example:
 |          HYPERPARAMETERS = {'dropout_rate': [0.1, 0.5, 0.9]}
 |
 |  Method resolution order:
 |      ModelBase
 |      torch.nn.modules.module.Module
 |      builtins.object
 |
 |  Readonly properties defined here:
 |
 |  criterion
 |      Get a criterion module for the model. Input/Output formats are defined for each task.
 |
 |  plugins
 |      List of plugins that are required for the model
 |
 |  predictor
 |      Get a predictor module for the model. Input/Output formats are defined for each task.
```

```
class PluginBase(builtins.object)
 |  Callbacks to modify the training behavior.
 |
 |  Methods defined here:
 |
 |  on_prediction_end(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the prediction ends.
 |
 |  on_prediction_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the prediction begins.
 |
 |  on_save_model(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
 |      Called when the trainer saves a pytorch model state.
 |
 |  on_save_onnx_end(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', onnx_model: bytes) -> bytes
 |      Called when the model is saved as onnx. Returns an updated onnx model binary.
 |
 |  on_save_onnx_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', nn_model: torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
 |      Called before ONNX tracing is performed. The returned nn_model will be used for tracing.
 |
 |  on_train_backward_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', loss: torch.Tensor) -> torch.Tensor
 |      Called when the train backward step begins. Returns a modified loss.
 |
 |  on_train_batch_end(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', batch: Tuple, batch_index: int) -> None
 |      Called when the train batch ends.
 |
 |  on_train_batch_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase', batch: Tuple, batch_index: int) -> Any
 |      Called when the train batch begins. Returns a modified batch.
 |
 |  on_train_end(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the training ends.
 |
 |  on_train_epoch_end(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the train epoch ends.
 |
 |  on_train_epoch_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the train epoch begins.
 |
 |  on_train_start(self, trainer: vision_model.trainer_callback_interface.TrainerCallbackInterface, model: 'ModelBase') -> None
 |      Called when the training begins.
 |      Notes:
 |          - Optimizers are configured after this callback is called.
```

## IO for Model, Criterion, and Predictor

Here is our current recommended input/output format for model, criterion and predictor.

Notations
```
ImageTensor: FP32 Tensor with shape (N, 3, H, W). The image format is RGB and the value range is [0-1].
BoxTensor: FP32 Tensor with shape [num_boxes, 4] that represents bounding boxes on an image. [x1, y1, x2, y2]. 0<=x1<=x2=1, and 0<=y1<=y2<=1.
LossTensor: FP32 Tensor with shape (1,)
ImageFeature: FP32 Tensor with shape (N, **).
TextFeature: FP32 Tensor with shape (N, **).
num_boxes: the number of boxes in an image.
num_classes: the number of classes in a dataset.
```

### Image Classification
```
Model(ImageTensor) -> ImageFeature
Criterion(ImageFeature, Targets) -> LossTensor
Predictor(ImageFeature) -> Tensor[N, num_classes]
```

### Object Detection
```
Model(ImageTensor) -> ImageFeature
Criterion(ImageFeature, Targets) -> LossTensor
Predictor(ImageFeature) -> (List[BoxTensor], List[Torch[num_boxes, num_classes]])  # Box and probabilities
```

### Image Classification with Text
```
Model(ImageTensor, TextTokens) -> (ImageFeature, TextFeature)  # TextFeature will be None if TextTokens is None
Criterion((ImageFeature, TextFeature), Targets) -> LossTensor
Predictor(ImageFeature) -> Tensor[N, num_classes]
```

### Object Detection with Text
```
Model(ImageTensor, TextTokens) -> (ImageFeature, TextFeature)
Criterion((ImageFeature, TextFeature), Targets) -> LossTensor
Predictor(ImageFeature) -> (List[BoxTensor], List[Torch[num_boxes, num_classes]])
```