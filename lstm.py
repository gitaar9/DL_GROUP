import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import Mode, MidiClassicMusic
from cross_validator import CrossValidator
from networks import BaseNet
from parallel_cnn_lstm import PretrainedLSTM
from util import format_filename


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    # mem = process.get_memory_info()[0] / float(2 ** 20)
    return str(process.memory_info())


# RNN Model (Many-to-One)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.add_module('lstm', self.lstm)

        # Fully connected layer 1
        self.fc1 = nn.Linear(hidden_size, 256)
        self.add_module('fc1', self.fc1)

        # Fully connected layer 2
        self.fc2 = nn.Linear(256, num_classes)
        self.add_module('fc2', self.fc2)

    def forward(self, x):
        # Put the input in the right order
        x = x.permute(0, 2, 1)

        # Set initial states <-- This might be unnecessary
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

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
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, dropout=0.5, pretrained=False,
                 **kwargs):
        # load the model
        self.model = LSTM(
            num_classes=18,
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        if pretrained:
            self.model.lstm = PretrainedLSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.model.fc2 = nn.Linear(256, num_classes)
        self.model.add_module('fc2', self.model.fc2)

        super().__init__(**kwargs)

    def get_data_loaders(self, batch_size, cv_cyle):
        """
        This function is overwritten because the LSTM expects data without channels(in contrast to conv nets),
        therefore the datasets should be constructed with unsqueeze=False.
        """
        train_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TRAIN, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle, unsqueeze=False),
            batch_size=batch_size,
            shuffle=True
        )
        print("Loaded train set\nCurrent memory: {}".format(memory_usage_psutil()))
        val_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.VALIDATION, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle, unsqueeze=False),
            batch_size=batch_size,
            shuffle=False
        )
        print("Loaded validation set\nCurrent memory: {}".format(memory_usage_psutil()))
        test_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TEST, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle, unsqueeze=False),
            batch_size=batch_size,
            shuffle=False
        )
        print("Loaded test set\nCurrent memory: {}".format(memory_usage_psutil()))
        return train_loader, val_loader, test_loader


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
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Use a pretrained model?.')

    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size, args.dropout, args.pretrain


if __name__ == '__main__':
    arguments = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Beethoven', 'Haendel',
                 'Bach', 'Chopin']

    file_name = format_filename("lstm_11", arguments)

    # Unpack the commandline arguments for use
    epochs, optimizer, num_layers, hidden_size, dropout, chunks, pretrain = arguments

    cv = CrossValidator(
        model_class=OurLSTM,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        verbose=False
    )

    cv.cross_validate()
