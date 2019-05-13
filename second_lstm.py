import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, AUSLAN
from networks import BaseNet


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Put the input in the right order
        x = x.permute(0, 2, 1)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, **kwargs):
        # load the model
        self.model = RNN(num_classes=num_classes, input_size=input_size, num_layers=num_layers, hidden_size=hidden_size)

        super().__init__(**kwargs)

    def get_data_loaders(self, train_batch_size, val_batch_size):
        # train_loader = DataLoader(
        #     MidiClassicMusic(folder_path="./data/midi_files_npy", train=True, slices=16, composers=self.composers, unsqueeze=False),
        #     batch_size=train_batch_size,
        #     shuffle=True
        # )
        # val_loader = DataLoader(
        #     MidiClassicMusic(folder_path="./data/midi_files_npy", train=False, slices=16, composers=self.composers, unsqueeze=False),
        #     batch_size=val_batch_size,
        #     shuffle=False
        # )
        train_loader = DataLoader(
            AUSLAN(train=True, unsqueeze=False),
            batch_size=train_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            AUSLAN(train=False, unsqueeze=False),
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

    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size


if __name__ == '__main__':
    epochs, num_layers, hidden_size = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn']  # , 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    lstm = OurLSTM(
        composers=composers,
        num_classes=96,
        epochs=100,
        train_batch_size=1,
        val_batch_size=1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=22,
        verbose=False
    )
    metrics = lstm.run()
    lstm.save_metrics("lstm_test_{}_{}_{}".format(epochs, num_layers, hidden_size), metrics)

