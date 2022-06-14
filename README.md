# Common interfaces for vision models.

This package provides common interfaces for vision models. 

Each Model must have two additional torch.nn.Module; criterion and predictor. A criterion is used in the training phase to calculate the loss from the model outputs. A predictor is used in test phase to get final prediction results from the model outputs. A Model must define criterion() and predictor() property to return a corresponding module. The expected input/output formats for model, criterion, predictor are defined for each task.

When a model requires advanced changes to training behavior, it can implement a Plugin class inheriting PluginBase. For the detail, please see the description of the PluginBase class.

## Install
```
pip install vision_model
```

## Examples

### Image Classification model example
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

Pleasee see [examples/ directory](./examples) for more practical examples.

### Trainer implementation example
```python
class ImageClassificationTrainer(TrainerCallbackInterface):
    # TrainerCallbackInterface implementations are omitted here.

    def train(model: ModelBase, dataloader, additional_plugins):
        plugins = PluginList(model.plugins + additional_plugins)  # PluginList has PluginBase interfaces.

        plugins.on_train_start(self, model)
        for epoch in range(self.num_epochs):
            plugins.on_train_epoch_start(self, model)
            for batch_index, batch in enumerate(dataloader):
                batch = plugins.on_train_batch_start(self, model, batch, batch_index)
                features = model(batch)
                loss = model.criterion(features)
                loss.backward()

                # Omitted: optimizer.step(), lr_scheduler.update() here

                plugins.on_train_batch_end(self, model, batch, batch_index)
            plugins.on_train_epoch_end(self, model)

        plugins.on_train_end(self, model)

    def validate(model: ModelBase, dataloader):
        results = []
        for batch_index, batch in enumerate(dataloader):
            features = model(batch)
            outputs = model.predictor(features)
            results.append(outputs)

        return torch.cat(results)
```

## Input/Output for Model, Criterion, and Predictor

Here is our current recommended input/output format for model, criterion and predictor.

Notations
```
ImageTensor: FP32 Tensor with shape (N, 3, H, W). The image format is RGB and the value range is [0-1].
BoxTensor: FP32 Tensor with shape [num_boxes, 4] that represents bounding boxes on an image. Each box is in relative coordinates [x1, y1, x2, y2]. 0<=x1<x2<=1 and 0<=y1<y2<=1.
LossTensor: FP32 Tensor with shape (1,)
ImageFeature: FP32 Tensor with shape (N, **).
TextFeature: FP32 Tensor with shape (N, **).
num_boxes: the number of boxes in an image.
num_classes: the number of classes in a dataset.
TextTokens: A list of tokenized texts.
```

### Image Classification
```
Model(ImageTensor) -> ImageFeature
Criterion(ImageFeature, Targets) -> LossTensor  # Targets is Tensor[N, num_classes] and should be numbers between 0 and 1
Predictor(ImageFeature) -> Tensor[N, num_classes]
```

### Object Detection
```
Model(ImageTensor) -> ImageFeature
Criterion(ImageFeature, Targets) -> LossTensor  # Targets is List[Tensor[N, 5]]. A box is represented as [class_id (int), x1, y1, x2, y2].
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

### Image Captioning
```
Model(ImageTensor, TextTokens) -> (ImageFeature, TextFeature)
Criterion((ImageFeature, TextFeature), Targets) -> LossTensor  # Targets has same format with TextTokens.
Predictor(ImageFeature) -> TextTokens
```
