import inspect
import time

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, Mode


class BaseNet:
    def __init__(self, epochs, composers, batch_size=100, optimizer='Adadelta', verbose=True, cv_cycle=0):
        """
        :param epochs: The amount of epochs this network will be trained for when run() is called
        :param composers: The names of the composers that should be loaded as dataset
        :param batch_size: The size of the train/validation/test batches
        :param optimizer: The optimizer used either Adadelta or Adam
        :param verbose: Print the loss after every train_batch when training on cpu
        :param cv_cycle: How many steps the dataset should be cycled for the cross-validation to work
        """
        self.epochs = epochs

        self.composers = composers
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders(batch_size, cv_cycle)
        self.loss_function = nn.CrossEntropyLoss()  # cross entropy works well for multi-class problems

        # optimizer: Adadelta or Adam
        if optimizer == "Adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters())
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise Exception('You misspelled the optimizer:(')

        # See if we use CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()
        self.verbose = verbose

    def get_data_loaders(self, batch_size, cv_cyle):
        print("Loading datasets")
        train_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TRAIN, slices=40, composers=self.composers,
                             cv_cycle=cv_cyle),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.VALIDATION, slices=40,
                             composers=self.composers, cv_cycle=cv_cyle),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TEST, slices=40, composers=self.composers,
                             cv_cycle=cv_cyle),
            batch_size=batch_size,
            shuffle=False
        )
        print("Done loading datasets")
        return train_loader, val_loader, test_loader

    def change_data_loaders(self, batch_size, cv_cycle):
        del self.train_loader
        del self.val_loader
        del self.test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders(batch_size, cv_cycle)

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def train(self):
        total_loss = 0
        self.model.train()
        if self.cuda_available:
            self.model.cuda()
        for i, data in enumerate(self.train_loader):
            X, y = data[0].to(self.device), data[1].to(self.device)
            # training step for single batch
            self.model.zero_grad()
            outputs = self.model(X)

            loss = self.loss_function(outputs, y)
            loss.backward()
            self.optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            if not self.cuda_available and self.verbose:
                print(total_loss / (i + 1))

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss

    def validate(self, data_loader):
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X)  # this get's the prediction from the network

                val_losses += self.loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        self.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
        return val_losses, precision, recall, f1, accuracy

    def run(self):
        start_ts = time.time()

        metrics = []
        batches = len(self.train_loader)
        val_batches = len(self.val_loader)
        test_batches = len(self.test_loader)
        print("batches: {}, val_batches: {}, test_batches: {}".format(batches, val_batches, test_batches))

        for epoch in range(self.epochs):
            total_loss = self.train()
            val_losses, precision, recall, f1, accuracy = self.validate(self.val_loader)
            _, _, _, _, test_accuracy = self.validate(self.test_loader)

            print(f"Epoch {epoch+1}/{self.epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
            self.print_scores(precision, recall, f1, accuracy, val_batches)
            print(f"\t{'test accuracy'.rjust(14, ' ')}: {sum(test_accuracy)/test_batches:.4f}")
            metrics.append((total_loss / batches, val_losses / val_batches, sum(precision) / val_batches,
                            sum(recall) / val_batches, sum(f1) / val_batches, sum(accuracy) / val_batches,
                            sum(test_accuracy) / test_batches))  # for plotting learning curve

        print(f"Training time: {time.time()-start_ts}s")
        return metrics

    def free(self):
        del self.model
        del self.train_loader
        del self.val_loader
        del self.test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def calculate_metric(metric_fn, true_y, pred_y):
        # multi class problems need to have averaging method
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y)

    @staticmethod
    def print_scores(p, r, f1, a, batch_size):
        # just an utility printing function
        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

    @staticmethod
    def save_metrics(name, metrics):
        with open(name, 'w') as f:
            f.write('training_loss,validation_loss,precision,recall,f1,accuracy,test_accuracy\n')
            for training_loss, validation_loss, precision, recall, f1, accuracy, test_accuracy in metrics:
                f.write("{},{},{},{},{},{},{}\n".format(training_loss, validation_loss, precision, recall, f1, accuracy, test_accuracy))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        for name, module in self.model.named_modules():
            if name == 'lstm':
                torch.save(module.state_dict(), path + "_only_lstm")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
