import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cross_validator import CrossValidator
from networks import BaseNet


class SinglePassCnnLstmModel(nn.Module):
    def __init__(self, cnn_pretrained, feature_extract,
                 lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout):
        super().__init__()
        # We build the convolution network for our model.
        self.cnn_model = self.build_cnn(cnn_pretrained, feature_extract, lstm_input_size)
        # We build a LSTM network for our model.
        self.lstm_model = nn.LSTM(lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout=dropout)

        self.add_module('cnn', self.cnn_model)
        self.add_module('lstm', self.lstm_model)

    def forward(self, inputs):
        """
        The data is divided into the input data for the CNN
        (size batch_size x 72 x 1600/n_chunks) and the hidden and cell
        state of the LSTM from the previous time step. Then it passes this
        data through the CNN-LSTM network.
        Args:
            inputs: tuple of next chunk of input data and output of lstm in t-1

        Returns:
            tuple of hidden and cell state of LSTM
        """
        inputs, (h_n, c_n) = inputs
        cnn_output = self.cnn_model(inputs)
        output, (h_n, c_n) = self.lstm_model(cnn_output.unsqueeze(0), (h_n, c_n))

        return output, (h_n, c_n)

    def build_cnn(self, cnn_pretrained, feature_extract, lstm_input_size):
        # TODO: try building our own simple convolution layers (just a couple)
        model = models.resnet18(pretrained=cnn_pretrained)  # TODO: use densenet
        # model = models.resnet50(pretrained=cnn_pretrained)
        # Change input layer to 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=12, stride=2, padding=3, bias=False)
        if feature_extract:
            self.freeze_all_layers()
        # Change output layer
        model.fc = nn.Linear(512, lstm_input_size)  # 512 for resnet18, 2048 for resnet50
        return model


class CnnLstmModel(nn.Module):
    def __init__(self, num_classes, input_size, cnn_pretrained, feature_extract,
                 lstm_input_size, lstm_hidden_size, num_lstm_layers, dropout):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn_lstm = SinglePassCnnLstmModel(cnn_pretrained=cnn_pretrained,
                                               feature_extract=feature_extract,
                                               lstm_input_size=lstm_input_size,
                                               lstm_hidden_size=lstm_hidden_size,
                                               num_lstm_layers=num_lstm_layers,
                                               dropout=dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.classifier = nn.Linear(256, num_classes)

        self.add_module('cnn_lstm', self.cnn_lstm)
        self.add_module('fc1', self.fc1)
        self.add_module('classifier', self.classifier)

    def forward(self, x):
        """
        Divides the input data into smaller chunks. Then passes sequentially
        the chunks of data trough the CNN-LSTM network, while the LSTM also
        takes as inputs the hidden_state of the last layer of the LSTM in the
        previous time step. Hopefully it will classify the composer.
        Args:
            x: input data

        Returns:
            The activations of the last fully connected layer.

        """
        # Set initial states # TODO: could be random initialization, instead of zeroes
        h = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device)
        c = torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(self.device)

        # We divide the input data into chunks, then run them through the cnn_lstm model 1 by 1
        n_chunks = 20
        for chunk in torch.chunk(x, n_chunks, 3):
            output, (h, c) = self.cnn_lstm((chunk, (h, c)))
        # TODO: dropout should be here?
        # Dropout over the output of the lstm
        output = output.squeeze(0)
        output = F.dropout(output, p=self.dropout, training=self.training)

        # The output of the last layer of the lstm goes into the first fully connected layer
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test different cnn-lstm models on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_lstm_layers', type=int, default=1,
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

    return args.epochs, args.num_lstm_layers, args.lstm_hidden_size, args.dropout, args.lstm_input_size


if __name__ == '__main__':
    epochs, num_lstm_layers, lstm_hidden_size, dropout, lstm_input_size = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']

    file_name = "cnn_lstm_test_precision8_{}_{}_{}_{}_{}".format(epochs,
                                                                 num_lstm_layers,
                                                                 lstm_input_size,
                                                                 lstm_hidden_size,
                                                                 dropout)

    cv = CrossValidator(
        model_class=OurCnnLstm,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout,
        lstm_input_size=lstm_input_size,
        verbose=False
    )

    cv.cross_validate()
