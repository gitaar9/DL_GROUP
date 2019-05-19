import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import BaseNet


# RNN Model (Many-to-One)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_size=(72, 1600)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, bias=False)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, bias=False)
        self.maxpool2 = nn.MaxPool2d(2)
        after_maxpool_height = int((((input_size[0] - 4) / 2 - 4) / 2))
        after_maxpool_width = int((((input_size[1] - 4) / 2 - 4) / 2))
        self.fc1 = nn.Linear(16 * after_maxpool_height * after_maxpool_width, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        flattened_x = x.view(x.size()[0], -1)
        x = self.fc1(flattened_x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout over the output of the lstm

        x = self.fc2(x)
        return x


class OurSimpleCNN(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        # load the model
        self.model = SimpleCNN(num_classes=num_classes)

        super().__init__(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The lstm layers.')
    parser.add_argument('--hidden_size', type=int, default=8,
                        help='The amount of blocks in every lstm layer.')

    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size


if __name__ == '__main__':
    epochs, num_layers, hidden_size = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    lstm = OurSimpleCNN(
        composers=composers,
        num_classes=len(composers),
        epochs=100,
        train_batch_size=100,
        val_batch_size=100,
        verbose=False
    )
    metrics = lstm.run()
    lstm.save_metrics("results/cnn_test_{}_{}_{}".format(epochs, num_layers, hidden_size), metrics)

