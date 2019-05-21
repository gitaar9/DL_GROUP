import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic
from networks import BaseNet


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.add_module('lstm', self.lstm)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.add_module('fc1', self.fc1)
        self.fc2 = nn.Linear(256, num_classes)
        self.add_module('fc2', self.fc2)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Put the input in the right order
        x = x.permute(0, 2, 1)

        # Set initial states <-- This might be unnecessary
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))

        # Forward propagate RNN
        x, _ = self.lstm(x, (h0, c0))
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout over the output of the lstm

        # The output of the lstm for the last time step goes into the first fully connected layer
        x = self.fc1(x[:, -1, :])
        x = F.relu(x)

        # Pass to the last fully connected layer (SoftMax)
        x = self.fc2(x)
        return x


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, dropout=0.5, **kwargs):
        # load the model
        self.model = RNN(
            num_classes=num_classes,
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout
        )

        super().__init__(**kwargs)

    def get_data_loaders(self, train_batch_size, val_batch_size):
        train_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy", train=True, slices=16, composers=self.composers, unsqueeze=False),
            batch_size=train_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy", train=False, slices=16, composers=self.composers, unsqueeze=False),
            batch_size=val_batch_size,
            shuffle=False
        )
        return train_loader, val_loader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The lstm layers.')
    parser.add_argument('--hidden_size', type=int, default=8,
                        help='The amount of blocks in every lstm layer.')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='The dropout rate after each lstm layer.')

    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size, args.dropout


if __name__ == '__main__':
    epochs, num_layers, hidden_size, dropout = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    lstm = OurLSTM(
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        train_batch_size=100,
        val_batch_size=100,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        verbose=False
    )
    metrics = lstm.run()
    lstm.save_metrics("results/lstm_test5_{}_{}_{}_{}".format(epochs, num_layers, hidden_size, dropout), metrics)

