import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_validator import CrossValidator
from datasets import Mode, MidiClassicMusic
from networks import BaseNet
from stupid_overwrites import DenseNet
from datetime import date
from util import format_filename
from parallel_cnn_lstm import PretrainedLSTM

class LSTM_CNN_model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM
        self.lstm = PretrainedLSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True,
                                   pretrained=True)
        self.add_module('lstm', self.lstm)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        #self.add_module('lstm', self.lstm)

        # DenseNet
        self.dense_net = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)
        self.add_module('dense_net', self.dense_net)

    def forward(self, input):

        # Get ready for putting input dataset into lstm! as [batch, sequence, elements]
        lstm_input = input.permute(0, 2, 1)

        # Set initial states <-- This might be unnecessary
        h0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(self.device)

        # LSTM layers
        lstm_activation, _ = self.lstm(lstm_input, (h0, c0))
        #lstm_activation = F.dropout(lstm_activation, p=self.dropout,
        #                            training=self.training)  # Dropout over the output of the lstm

        lstm_output = lstm_activation

        # Get ready for DenseNet!
        densenet_activation = lstm_output.unsqueeze(1)

        # DenseNet layers
        results = self.dense_net(densenet_activation)

        return results


class Our_lstm_cnn(BaseNet):
    def __init__(self, num_classes=10, input_size=72, hidden_size=8, num_layers=1, dropout=0, **kwargs):
        self.model = LSTM_CNN_model(
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
    parser = argparse.ArgumentParser(description='Test some lstm_cnn model on the midi database :P')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The lstm layers.')
    parser.add_argument('--hidden_size', type=int, default=8,
                        help='The amount of blocks in every lstm layer.')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='The dropout rate after each lstm layer.')
    parser.add_argument('--block_config', type=int, default=[2, 2], nargs='+',
                        help='The configuration of the dense blocks.')
    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size, args.dropout, args.block_config


if __name__ == '__main__':
    epochs, num_layers, hidden_size, dropout, block_config = parse_arguments()
    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']

    block_config_string = '(' + ','.join([str(i) for i in block_config]) + ')'
    file_name = "lstm_cnn_test_precision8_{}_{}_{}_{}_{}".format(epochs, num_layers, hidden_size, dropout, block_config_string)
    file_name += date.today().strftime("_%b_%-d_%H")
    #filename = format_filename('lstm_cnn_test_precision8',)

    cv = CrossValidator(
        model_class=Our_lstm_cnn,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=25,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=0,
        verbose=False  # If I want to print all the results during learning -> True
    )

    cv.cross_validate()
