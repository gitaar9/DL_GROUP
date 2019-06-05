import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_validator import CrossValidator
from datasets import Mode, MidiClassicMusic
from networks import BaseNet
from util import format_filename


# RNN Model (Many-to-Less)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, n_chunks):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_chunks = n_chunks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.add_module('lstm', self.lstm)

        # Fully connected layer 1
        self.fc1 = nn.Linear(hidden_size * self.n_chunks, 256)
        self.add_module('fc1', self.fc1)

        # Fully connected layer 2
        self.fc2 = nn.Linear(256, num_classes)
        self.add_module('fc2', self.fc2)

    def forward(self, x):
        # Put the input in the right order
        x = x.permute(0, 2, 1)

        # Set initial states <-- This might be unnecessary
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        outputs = []
        # Forward propagate RNN
        for chunk in torch.chunk(x, self.n_chunks, 1):
            output, (h, c) = self.lstm(chunk, (h, c))
            outputs.append(output[:, -1, :])

        x = torch.cat(outputs, dim=1)  # Concatenate all outputs
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout over the output of the lstm

        # The output of the lstm for the last time step goes into the first fully connected layer
        x = self.fc1(x)
        x = F.relu(x)

        # Pass to the last fully connected layer (SoftMax)
        x = self.fc2(x)
        return x


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, dropout=0.5, n_chunks=10,
                 pretrained=False, pretrained_model_name=None, **kwargs):
        # load the model
        self.model = LSTM(
            num_classes=18,
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            n_chunks=n_chunks
        )

        if pretrained and pretrained_model_name:
            self.load_model('pretrained_models/{}'.format(pretrained_model_name))
        self.model.fc2 = nn.Linear(256, num_classes)
        self.model.add_module('fc2', self.fc2)

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
        val_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.VALIDATION, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle, unsqueeze=False),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TEST, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle, unsqueeze=False),
            batch_size=batch_size,
            shuffle=False
        )
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
    parser.add_argument('--chunks', type=int, default=10,
                        help='How many chunks to make of the input sequence.')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='Please decide which optimizer you want to use: Adam or Adadelta')
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Use a pretrained model?.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='The name of the pretrained model to load.')

    args = parser.parse_args()

    return args.epochs, args.optimizer, args.num_layers, args.hidden_size, args.dropout, args.chunks, args.pretrain, args.model_name


if __name__ == '__main__':
    arguments = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']

    file_name = format_filename("advanced_lstm_test", ("precision8", ) + arguments)

    # Unpack the commandline arguments for use
    epochs, optimizer, num_layers, hidden_size, dropout, chunks, pretrain, model_name = arguments

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
        n_chunks=chunks,
        optimizer=optimizer,
        pretrained=pretrain,
        pretrained_model_name=model_name,
        verbose=False
    )

    cv.cross_validate()
