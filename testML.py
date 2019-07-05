import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, Mode

composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Haendel', 'Beethoven', 'Bach', 'Chopin']
run_type = 'composers'

train = MidiClassicMusic(folder_path="./data/midi_files_npy_8_40",composers=composers, add_noise=False,
                             run_type=run_type, unsqueeze=False)
print("loaded dataset")
inputs = []
labels = []
index = 0

oldIndexes = [0]
for composer in composers:
    for i in np.arange(np.shape(train.data_array[index])[0]):
        if (i%100 == 0):
            print(i/np.shape(train.data_array[index])[0])
        X, y = train.__getitem__(np.sum(oldIndexes) + i)
        X = X.numpy()
        x = []
        for i in np.arange(np.shape(X)[0]):
            for j in np.arange(np.shape(X)[1]):
                x.append(X[i, j])
        inputs.append(x)
        labels.append(y)
    oldIndexes.append(i)
    index += 1

print(np.shape(labels))
unique, counts = np.unique(labels, return_counts=True)

print(np.asarray((unique, counts)).T)

x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1)

print("load classifier")
clf = MLPClassifier(verbose=True, early_stopping=True, validation_fraction=0.25)

scores = []
for i in np.arange(4):
    print("load classifier")
    clf = MLPClassifier(verbose=True, early_stopping=True, validation_fraction=0.25)
    print("start model fit: " + str(i))
    clf = clf.fit(x_train, y_train)

    test_score = clf.score(x_test, y_test)
    scores.append(test_score)

file_name = "cross_validated_results/MLP_precision8_2"

np.savetxt(file_name, scores)

