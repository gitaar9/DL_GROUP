import inspect
import time

import torch
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic
from stupid_overwrites import densenet121


class BaseNet:
    def __init__(self, epochs, train_batch_size=100, val_batch_size=100, optimizer='Adadelta'):
        #params you need to specify:
        self.epochs = epochs
        # put your data loader here
        self.train_loader, self.val_loader = self.get_data_loaders(train_batch_size, val_batch_size)
        self.loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

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

    @staticmethod
    def get_data_loaders(train_batch_size, val_batch_size):
        composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']
        train_loader = DataLoader(MidiClassicMusic(folder_path="./data/midi_files_npy", train=True, slices=8, composers=composers), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(MidiClassicMusic(folder_path="./data/midi_files_npy", train=False, slices=8, composers=composers), batch_size=val_batch_size, shuffle=False)
        return train_loader, val_loader

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

            if not self.cuda_available:
                print(total_loss/(i+1))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss
        
    def validate(self):
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                X, y = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X) # this get's the prediction from the network

                val_losses += self.loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
                
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
        print("batchs: {}, val_batches: {}".format(batches, val_batches))
        
        for epoch in range(self.epochs):
            total_loss = self.train()
            val_losses, precision, recall, f1, accuracy = self.validate()
            
            print(f"Epoch {epoch+1}/{self.epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
            self.print_scores(precision, recall, f1, accuracy, val_batches)
            metrics.append((total_loss/batches, val_losses/val_batches, sum(precision)/val_batches, sum(recall)/val_batches, sum(f1)/val_batches, sum(accuracy)/val_batches)) # for plotting learning curve
        
        print(f"Training time: {time.time()-start_ts}s")
        return metrics

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
            f.write('training_loss,validation_loss,precision,recall,f1,accuracy\n')
            for training_loss, validation_loss, precision, recall, f1, accuracy in metrics:
                f.write("{},{},{},{},{},{}\n".format(training_loss, validation_loss, precision, recall, f1, accuracy))


class OurResNet(BaseNet):
    def __init__(self, num_classes=10, pretrained=True, feature_extract=False, **kwargs):
        # load the model
        self.model = models.resnet50(pretrained=pretrained)
        # Change input layer to 1 channel
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=12, stride=2, padding=3, bias=False)
        if feature_extract:
            self.freeze_all_layers()
        # Change output layer
        self.model.fc = nn.Linear(2048, num_classes)  # 512 for resnet18, 2048 for resnet50

        super().__init__(**kwargs)


class OurDenseNet(BaseNet):
    def __init__(self, num_classes=10, pretrained=True, feature_extract=False, **kwargs):
        # load the model
        self.model = densenet121(pretrained=pretrained)
        if feature_extract:
            self.freeze_all_layers()
        self.model.classifier = nn.Linear(1024, num_classes)

        super().__init__(**kwargs)


if __name__ == '__main__':
    dense = OurResNet(num_classes=10, pretrained=False, epochs=100, train_batch_size=50)
    #print(dense.model.conv1)
    metrics = dense.run()
