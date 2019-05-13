import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, AUSLAN
from networks import BaseNet


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_dim=100, num_lstm_layers=1, batch_size):

        super().__init__()
        self.num_lstm_layers = num_lstm_layers
        self.nb_lstm_units = hidden_dim

        # sequential model
        self.lstm = nn.LSTM(input_size, hidden_dim, num_lstm_layers, dropout=.5)
        self.add_module('lstm', self.lstm)

        self.dropout = nn.Dropout(.8)
        self.add_module('dropout', self.dropout)

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.add_module('fc', self.fc)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.num_lstm_layers, 1, self.nb_lstm_units)
        hidden_b = torch.randn(self.num_lstm_layers, 1, self.nb_lstm_units)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return hidden_a, hidden_b

    def forward(self, x):
        self.hidden = self.init_hidden()
        x = x.permute(2, 0, 1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(lstm_out[-1])
        predicted = self.fc(out)
        return predicted


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        # load the model
        self.model = LSTMModel(num_classes=num_classes, input_size=22, num_lstm_layers=1)

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


if __name__ == '__main__':
    # composers = ['Brahms', 'Mozart']  #, 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    lstm = OurLSTM(composers=[], num_classes=96, epochs=100, train_batch_size=1, val_batch_size=1, verbose=False)
    metrics = lstm.run()
