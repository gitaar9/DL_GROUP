import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_validator import CrossValidator
from datasets import Mode, MidiClassicMusic
from networks import BaseNet
from stupid_overwrites import DenseNet
from util import format_filename


class PretrainedDenseNet(DenseNet):

    def __init__(self, *args, **kwargs):
        num_classes = kwargs.pop('num_classes', 11)
        pretrained = kwargs.pop('pretrained', False)
        kwargs['num_classes'] = 18
        super().__init__(*args, **kwargs)

        # Load pretrained network
        if pretrained:
            self.load_state_dict(torch.load('pretrained_models/densenet_test_precision8_75_adadelta'))

        self.classifier = nn.Linear(self.output_size, num_classes)


class PretrainedDenseNetWithoutFC(PretrainedDenseNet):
    """
    Overwrite of Densenet that doesnt have a fully connected layer, so this just always has outputsize 1024.
    """
    def forward(self, x):
        # Overwrite densenet to not use its classifier
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


class PretrainedLSTM(nn.LSTM):
    """This only works with 2 layers of 256 nodes"""
    def __init__(self, *args, **kwargs):
        pretrained = kwargs.pop('pretrained', False)
        if pretrained:
            nl = kwargs['num_layers']
            hs = kwargs['hidden_size']
        super().__init__(*args, **kwargs)

        # Load pretrained network
        if pretrained:
            path = 'pretrained_models/advanced_lstm_test_precision8_75_Adadelta_{}_{}_0.8_20_only_lstm'.format(nl, hs)
            try:
                self.load_state_dict(torch.load(path))
            except:
                raise Exception('Didnt find pretrained model')


# Parallel CNN LSTM Model from the Acoustic Scenes Classification paper(with densenet though)
class ParallelCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The LSTM layers
        self.lstm = PretrainedLSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                   batch_first=True, pretrained=True)
        self.add_module('lstm', self.lstm)

        # Dense net
        self.dense_net = PretrainedDenseNetWithoutFC(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                                                     num_classes=num_classes, pretrained=True)
        self.add_module('dense_net', self.dense_net)

        # Fully connected layer 1
        self.fc1 = nn.Linear(hidden_size + self.dense_net.output_size, 512)
        self.add_module('fc1', self.fc1)

        # Fully connected layer 2
        self.fc2 = nn.Linear(512, 512)
        self.add_module('fc2', self.fc2)

        # Fully connected layer 3
        self.fc3 = nn.Linear(512, num_classes)
        self.add_module('fc3', self.fc3)

    def forward(self, input):
        # LSTM FORWARDING PART
        lstm_activation = input.permute(0, 2, 1)  # Put the input in the right order (batch, sequence, elements)

        # Set initial states <-- This might be unnecessary
        h0 = torch.zeros(self.num_layers, lstm_activation.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, lstm_activation.size(0), self.hidden_size).to(self.device)

        # Forward propagate RNN
        lstm_activation, _ = self.lstm(lstm_activation, (h0, c0))
        lstm_activation = F.dropout(lstm_activation, p=self.dropout, training=self.training)  # Dropout over the output of the lstm

        lstm_activation = lstm_activation[:, -1, :]  # Use output for last timestep only

        # DENSENET FORWARDING PART
        densenet_activation = input.unsqueeze(1)
        densenet_activation = self.dense_net(densenet_activation)

        # FULLY CONNECTED PART
        x = torch.cat((lstm_activation, densenet_activation), dim=1)  # Concatenate the two outputs

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # Pass to the last fully connected layer (SoftMax)
        x = self.fc3(x)
        return x


class OurParallelCNNLSTM(BaseNet):
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, dropout=0.5, **kwargs):
        # load the model
        self.model = ParallelCNNLSTM(
            num_classes=num_classes,
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout
        )

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

    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size, args.dropout


if __name__ == '__main__':
    arguments = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Beethoven', 'Haendel',
                 'Bach', 'Chopin']

    file_name = format_filename("parallel_cnn_lstm_11", arguments)

    epochs, num_layers, hidden_size, dropout = arguments
    cv = CrossValidator(
        model_class=OurParallelCNNLSTM,
        file_name=file_name,
        folder='final_results',
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
