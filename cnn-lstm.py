import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cross_validation_datasets import Mode, MidiClassicMusic
from cross_validator import CrossValidator
from networks import BaseNet

# from lstm import OurLSTM
import lstm
import networks


class CnnLstmModel(nn.Module):
    def __init__(self, num_classes, input_size, cnn_pretrained, feature_extract,
                 lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN model
        self.cnn_model = self.build_cnn(cnn_pretrained, feature_extract, lstm_input_size)

        # The LSTM layers
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout=dropout, batch_first=True)
        self.add_module('lstm', self.lstm)

        # Fully connected layer 1
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.add_module('fc1', self.fc1)

        # Fully connected layer 2
        self.fc2 = nn.Linear(256, num_classes)
        self.add_module('fc2', self.fc2)

    def build_cnn(self, cnn_pretrained, feature_extract, lstm_input_size):
        model = models.resnet50(pretrained=cnn_pretrained)
        # Change input layer to 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=12, stride=2, padding=3, bias=False)
        if feature_extract:
            self.freeze_all_layers()
        # Change output layer
        model.fc = nn.Linear(2048, lstm_input_size)  # 512 for resnet18, 2048 for resnet50
        return model

    def forward(self, x):
        # Put the input in the right order
        # x = x.permute(0, 2, 1)  # TODO:needed here?

        # Forward pass through ResNet
        x = self.cnn_model.conv1
        x = self.cnn_model.conv1(x)
        x = self.cnn_model.bn1(x)
        x = self.cnn_model.relu(x)
        x = self.cnn_model.maxpool(x)

        x = self.cnn_model.layer1(x)
        x = self.cnn_model.layer2(x)
        x = self.cnn_model.layer3(x)
        x = self.cnn_model.layer4(x)

        x = self.cnn_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.cnn_model.fc(x)

        # Forward pass through LSTM

        # Set initial states <-- This might be unnecessary
        h0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device))
        c0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device))

        output, (hn, cn) = self.lstm(x, (h0, c0))
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout over the output of the lstm

        # The output of the lstm for the last time step goes into the first fully connected layer
        # x = self.fc1(x[:, -1, :])
        x = self.fc1(hn)
        x = F.relu(x)

        # Pass to the last fully connected layer (SoftMax)
        x = self.fc2(x)
        return x


class OurCnnLstm(BaseNet):
    def __init__(self, num_classes=10, input_size=72, cnn_pretrained=True, feature_extract=False,
                 lstm_input_size = 72, lstm_hidden_size=8, num_lstm_layers=1, dropout=0.5, **kwargs):
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

    # composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    composers = ['Brahms', 'Mozart', 'Schubert']

    file_name = "lstm_test_precision8_{}_{}_{}_{}".format(epochs, num_layers, hidden_size, dropout)

    cv = CrossValidator(
        model_class=OurCnnLstm,
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