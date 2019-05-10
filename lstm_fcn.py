import torch
import torch.nn as nn

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
        lstm_out, self.hidden = self.lstm(x)
        predicted = self.linear(lstm_out[-1])
        return predicted


class OurLSTM(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        # load the model
        self.model = LSTMModel(num_classes=num_classes, input_size=(1600, 72,))

        super().__init__(**kwargs)


if __name__ == '__main__':
    lstm = OurLSTM(num_classes=10, epochs=100, train_batch_size=30, val_batch_size=30)
    metrics = lstm.run()
    print(metrics)