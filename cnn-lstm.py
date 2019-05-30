import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import OrderedDict

from cross_validation_datasets import Mode, MidiClassicMusic
from cross_validator import CrossValidator
from networks import BaseNet

# from lstm import OurLSTM
import lstm
import networks


class SinglePassCnnLstmModel(nn.Module):
    def __init__(self, cnn_pretrained, feature_extract,
                 lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout):
        super().__init__()

        self.cnn_model = self.build_cnn(cnn_pretrained, feature_extract, lstm_input_size)
        # TODO: work to be done to fix the sequence dimension..
        self.lstm_model = nn.LSTM(lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout=dropout, batch_first=True)

        # self.classifier = nn.Linear(256, num_classes)
        # self.classifier.add_module('fc2', nn.Linear(256, num_classes))

        self.model = nn.Sequential(OrderedDict([
            ('cnn', self.cnn_model),
            ('lstm', self.lstm_model),
        ]))
        # self.lstm_fc1 =nn.Linear(lstm_hidden_size, 256)
        # self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs):
        inputs, (h_n, c_n) = inputs
        cnn_output = self.cnn_model(inputs)
        output, (h_n, c_n) = self.lstm_model(cnn_output, (h_n, c_n))
        return (h_n, c_n)

    # From densenet forward function:
    # def forward(self, x):
    #     features = self.features(x)
    #     out = F.relu(features, inplace=True)
    #     out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
    #     out = self.classifier(out)
    #     return out

    def build_cnn(self, cnn_pretrained, feature_extract, lstm_input_size):
        # TODO: try building our own simple convolution layers (just a couple)
        model = models.resnet18(pretrained=cnn_pretrained)  # TODO: use densenet
        # model = models.resnet50(pretrained=cnn_pretrained)
        # Change input layer to 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=12, stride=2, padding=3, bias=False)
        if feature_extract:
            self.freeze_all_layers()
        # Change output layer
        model.fc = nn.Linear(2048, lstm_input_size)  # 512 for resnet18, 2048 for resnet50
        return model


class CnnLstmModel(nn.Module):
    def __init__(self, num_classes, input_size, cnn_pretrained, feature_extract,
                 lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn_lstm = SinglePassCnnLstmModel(num_classes,
                                               input_size,
                                               cnn_pretrained,
                                               feature_extract,
                                               lstm_input_size,
                                               lstm_hidden_size,
                                               num_lstm_layers,
                                               dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.model = nn.Sequential(OrderedDict([
            ('cnn_lstm', self.cnn_lstm),
            ('fc1', self.fc1),
            ('classifier', self.classifier),
        ]))

    def forward(self, x):
        # Set initial states <-- This might be unnecessary
        h = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device))
        c = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device))

        n_chunks = 20
        for chunk in torch.chunk(x, n_chunks, 0):
            h, c = self.cnn_lstm((chunk, (h, c)))
        # TODO: dropout should be here?
        output = F.dropout(h, p=self.dropout, training=self.training)  # Dropout over the output of the lstm
        # The output of the lstm goes into the first fully connected layer
        output = self.fc1(output)
        output = F.relu(output)
        # Pass to the last fully connected layer (SoftMax)
        output = self.classifier(output)
        return output


class OurCnnLstm(BaseNet):
    def __init__(self, num_classes=10, input_size=72, cnn_pretrained=False,
                 feature_extract=False, lstm_input_size=512, lstm_hidden_size=256,
                 num_lstm_layers=1, dropout=0.5, **kwargs):
        # load the model
        self.model = CnnLstmModel(
            num_classes=num_classes,
            input_size=input_size,
            cnn_pretrained=cnn_pretrained,
            feature_extract=feature_extract,
            lstm_input_size=lstm_input_size,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
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
    parser = argparse.ArgumentParser(description='Test different cnn-lstm models on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of lstm layers.')
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help='The amount of blocks in every lstm layer.')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='The dropout rate after each lstm layer.')
    # parser.add_argument('--cnn_pretrained', type=bool, default=False,
    #                     help='If the CNN network uses pretrained weights.')
    # parser.add_argument('--feature_extract', type=bool, default=False,
    #                     help='If the CNN freezes the weights so no more training.')
    parser.add_argument('--lstm_input_size', type=int, default=512,
                        help='The output size of the CNN, as well as the input size of the LSTM.')
    args = parser.parse_args()

    return args.epochs, args.num_layers, args.hidden_size, args.dropout, args.lstm_input_size


if __name__ == '__main__':
    epochs, num_layers, lstm_hidden_size, dropout, lstm_input_size = parse_arguments()

    # composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    composers = ['Brahms', 'Mozart', 'Schubert']

    file_name = "lstm_test_precision8_{}_{}_{}_{}_{}".format(epochs, num_layers, lstm_hidden_size, dropout, lstm_input_size)

    cv = CrossValidator(
        model_class=OurCnnLstm,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout,
        lstm_input_size=lstm_input_size,
        verbose=False
    )

    cv.cross_validate()
