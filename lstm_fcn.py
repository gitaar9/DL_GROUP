import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic
from networks import BaseNet


class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_dim=100, num_lstm_layers=1):

        super().__init__()

        # sequential model
        self.lstm = nn.LSTM(input_size, hidden_dim, num_lstm_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden = None

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        x = x.permute(2, 0, 1)
        lstm_out, self.hidden = self.lstm(x)
        predicted = self.fc(lstm_out[-1])
        return predicted


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        # load the model
        self.model = LSTMModel(num_classes=num_classes, input_size=72)

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


if __name__ == '__main__':
    composers = ['Brahms', 'Mozart']  #, 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
    lstm = OurLSTM(composers=composers, num_classes=2, epochs=100, train_batch_size=10, val_batch_size=10)
    metrics = lstm.run()
    print(metrics)
