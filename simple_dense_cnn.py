import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _Transition

from networks import BaseNet


# Simple CNN implementation
class SimpleDenseCNN(nn.Module):
    def __init__(self, growth_rate=32, block_config=(2, 2), num_init_features=64, bn_size=4,
                 drop_rate=0, num_classes=1000):
        super().__init__()
        self.drop_rate=drop_rate
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=12, stride=2, padding=3, bias=False)),
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
            # + avg pooling
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.fc1 = nn.Linear(25344, 512)  # 25344 is the amount of values after the convolution
        self.fc2 = nn.Linear(512, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add the modules for train()/eval() to work
        for name, module in [('features', self.features), ('fc1', self.fc1), ('fc2', self.fc2)]:
            self.add_module(name, module)

    def forward(self, x):
        features = self.features(x)
        x = F.relu(features, inplace=True)

        # Flatten the output of features to fit in a fully connected layer
        x = x.view(x.size()[0], -1)
        print(x.shape)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        # First fully connected layer with relu and dropout
        x = self.fc1(x)
        x = F.relu(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        # The softmax output layer
        x = self.fc2(x)
        return x


class OurSimpleDenseCNN(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        # load the model
        self.model = SimpleDenseCNN(
            num_classes=num_classes
        )

        super().__init__(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')

    args = parser.parse_args()

    return args.epochs


if __name__ == '__main__':
    epochs = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    net = OurSimpleDenseCNN(
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        train_batch_size=50,
        val_batch_size=50,
        verbose=False
    )
    metrics = net.run()
    filename = "results/dense_cnn_test1_{}".format(epochs)
    net.save_metrics(filename, metrics)

