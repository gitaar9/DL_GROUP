import argparse

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
    def __init__(self, num_classes=10, pretrained=False, feature_extract=False, pretrained_model_name=None, **kwargs):
        # load the model
        self.model = densenet121(pretrained=False)

        if pretrained and pretrained_model_name:
            self.load_model('pretrained_models/{}'.format(pretrained_model_name))

        if feature_extract:
            self.freeze_all_layers()

        self.model.classifier = nn.Linear(1024, num_classes)

        super().__init__(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a densenet.')
    parser.add_argument('--epochs', type=int, default=400,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--feature_extract', default=False, action='store_true',
                        help='When this argument is supplied feature extraction instead of fine-tuing is used.')
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Use a pretrained model?.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='The name of the pretrained model to load.')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='Please decide which optimizer you want to use: Adam or Adadelta')

    args = parser.parse_args()

    return args.epochs, args.optimizer, args.feature_extract, args.pretrain, args.model_name


if __name__ == '__main__':
    arguments = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    file_name = format_filename("densenet_test", ("precision8",) + arguments)

    epochs, optimizer, feature_extract, pretrain, model_name = arguments
    cv = CrossValidator(
        model_class=OurDenseNet,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        feature_extract=feature_extract,
        pretrained=True,
        pretrained_model_name=model_name,
        optimizer=optimizer,
        verbose=False
    )
    cv.cross_validate()
