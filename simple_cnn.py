import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import BaseNet


# Simple CNN implementation
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, use_batch_norm=True, dropout=0.4, kernel_size=5, input_size=(72, 1600)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(2)

        after_maxpool_height = int((((input_size[0] - 4) / 2 - 4) / 2))
        after_maxpool_width = int((((input_size[1] - 4) / 2 - 4) / 2))
        self.fc1 = nn.Linear(16 * after_maxpool_height * after_maxpool_width, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        # Add the modules for train()/eval() to work
        for name, module in [('conv1', self.conv1), ('bn1', self.bn1), ('maxpool1', self.maxpool1),
                             ('conv2', self.conv2), ('bn2', self.bn2), ('maxpool2', self.maxpool2),
                             ('fc1', self.fc1), ('fc2', self.fc2)]:
            self.add_module(name, module)

    def forward(self, x):
        # First convolutional layer conv -> batch_norm -> relu -> maxpool
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        # Second convolutional layer conv -> batch_norm -> relu -> maxpool
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Flatten the output of conv2 to fit in a fully connected layer
        x = x.view(x.size()[0], -1)
        # Dropout after second pooling layer 30% in original paper
        x = F.dropout(x, p=self.dropout - 0.1 if self.dropout > 0.1 else 0, training=self.training)

        # First fully connected layer with relu and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after first fully connected layer 40%

        # The softmax output layer
        x = self.fc2(x)
        return x


class OurSimpleCNN(BaseNet):
    def __init__(self, num_classes=10, use_batch_norm=True, dropout=0.4, kernel_size=5, **kwargs):
        # load the model
        self.model = SimpleCNN(
            num_classes=num_classes,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            kernel_size=kernel_size
        )

        super().__init__(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--use_bn', default=False, action='store_true',
                        help='When this argument is supplied batch normalization is used in the convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='The dropout percentage used after the first fully-connected layer, this-0.1 is used as '
                             'dropout percentage for the last conv layer.')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='The amount of epochs that the model will be trained.')

    args = parser.parse_args()

    return args.epochs, args.use_bn, args.dropout, args.kernel_size


if __name__ == '__main__':
    epochs, use_bn, dropout, kernel_size = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    net = OurSimpleCNN(
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        train_batch_size=100,
        val_batch_size=100,
        verbose=False,
        use_batch_norm=use_bn,
        dropout=dropout,
        kernel_size=kernel_size
    )
    metrics = net.run()
    net.save_metrics("results/cnn_test3_{}_{}_{}_{}".format(epochs, "bn" if use_bn else "", dropout, kernel_size), metrics)

