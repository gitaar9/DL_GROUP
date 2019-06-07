import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _Transition

from cross_validator import CrossValidator
from networks import BaseNet
from datasets import MidiClassicMusic


class SimpleDense(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(SimpleDense, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.output_size = num_features

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class OurSimpleDense(BaseNet):
    def __init__(self, num_classes=10, dropout=0.0, block_config=None, **kwargs):
        self.model = SimpleDense(
            num_classes=num_classes,
            drop_rate=dropout,
            block_config=block_config
        )

        super().__init__(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='The dropout percentage used after the first fully-connected layer, this-0.1 is used as '
                             'dropout percentage for the last conv layer.')
    parser.add_argument('--block_config', type=int, default=[2, 2], nargs='+',
                        help='The configuration of the dense blocks.')

    args = parser.parse_args()

    return args.epochs, args.dropout, args.block_config


if __name__ == '__main__':
    composers = ['Brahms', 'Bach']

    # run_type specifies the type of classification, default should be 'composers'
    # other options are 'era' or 'country' as year does not work yet
    run_type = 'era'
    midi = MidiClassicMusic(composers=composers, run_type=run_type)
    num_classes = len(midi.classes)

    dropout = 0.6
    epochs = 10
    block_config = [2, 2]
    block_config_string = '(' + ','.join([str(i) for i in block_config]) + ')'
    file_name = "dense_test_precision8_{}_{}_{}".format(epochs, dropout, block_config_string)

    cv = CrossValidator(
        model_class=OurSimpleDense,
        file_name=file_name,
        composers=composers,
        run_type=run_type,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=100,
        verbose=False,
        dropout=dropout,
        block_config=block_config
    )

    cv.cross_validate()
