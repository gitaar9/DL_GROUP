import argparse

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, Mode
from res_densenet import OurDenseNet


class CurrentNetwork(OurDenseNet):
    def validate(self, data_loader):
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        self.model.eval()
        if self.cuda_available:
            self.model.cuda()

        hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X)  # this get's the prediction from the network
                for output in outputs:
                    max = -9999
                    max_idx = -1
                    for idx, node in enumerate(output):
                        if node > max:
                            max = node
                            max_idx = idx
                    hist[max_idx] += 1

                val_losses += self.loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        self.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
        return val_losses, precision, recall, f1, accuracy, np.array(hist)


def parse_arguments():
    parser = argparse.ArgumentParser(description='FILL IN LATER')
    parser.add_argument('filename', type=str,
                        help='The base filename of the files we want to read in when using amount_of_files > 1.'
                        'Otherwise just full filenames of the files.')
    parser.add_argument('--amount_of_files', type=int, default=1, help='The amount of runs we did.')

    args = parser.parse_args()
    return args.amount_of_files, args.filename


if __name__ == '__main__':
    amount_of_files, filename = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Beethoven', 'Haendel',
                 'Bach', 'Chopin']

    net = CurrentNetwork(epochs=1, save_path="-", num_classes=11, batch_size=100, composers=['Brahms'])

    composers_accuracies = []
    for label, composer in enumerate(composers):
        print("Getting accuracy for {}".format(composer))
        acc = 0
        test_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TEST, slices=40,
                             composers=[composer], cv_cycle=0, always_same_label=label),
            batch_size=10,
            shuffle=False
        )
        hist = np.zeros((1, 11))
        for network_name in ["best_models/{}_run{}".format(filename, file_number) for file_number in range(amount_of_files)]:
            net.load_model(network_name)
            _, _, _, _, acc_list, ret_hist = net.validate(test_loader)
            hist += ret_hist
            acc += sum(acc_list)/len(test_loader)
        hist /= 4
        print('all answers: ', hist)
        composers_accuracies.append(acc / 4)

    print('\n')
    for name, accuracy in zip(composers, composers_accuracies):
        print("{}: {}".format(name, accuracy))

    print("Overall accuracy {}".format(sum(composers_accuracies)/len(composers)))
