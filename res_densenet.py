from torch import nn
from torchvision import models as models

from cross_validator import CrossValidator
from networks import BaseNet
from stupid_overwrites import densenet121
from util import format_filename


class OurResNet(BaseNet):
    def __init__(self, num_classes=10, pretrained=False, feature_extract=False, **kwargs):
        # load the model
        self.model = models.resnet50(pretrained=pretrained)
        # Change input layer to 1 channel
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=12, stride=2, padding=3, bias=False)
        if feature_extract:
            self.freeze_all_layers()
        # Change output layer
        self.model.fc = nn.Linear(2048, num_classes)  # 512 for resnet18, 2048 for resnet50

        super().__init__(**kwargs)


class OurDenseNet(BaseNet):
    def __init__(self, num_classes=10, pretrained=False, feature_extract=False, **kwargs):
        # load the model
        self.model = densenet121(pretrained=pretrained)
        if feature_extract:
            self.freeze_all_layers()
        self.model.classifier = nn.Linear(1024, num_classes)

        super().__init__(**kwargs)


if __name__ == '__main__':
    epochs = 400
    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    file_name = format_filename("densenet_test", ("precision8", epochs, "adadelta"))
    cv = CrossValidator(
        model_class=OurDenseNet,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        pretrained=False,
        verbose=False
    )
    cv.cross_validate()
