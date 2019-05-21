# DL_GROUP
Composer prediction based on midi files

converter.py:
This file reads in the midi files and stores them in npy files that can be easily read in while training a network.
There are several options of how to do this: precision, lowest/highest octave.
At the moment a song is sliced in parts of 100 timesteps wide, this should be refactored.

datasets.py:
Implements the classical midi dataset dataset, when training it will provide songs from all composers with equal chance.
This is important because otherwise the network would for instance see much more Mozart than Liszt.
While validating the dataset works more intuitively.

main.py:
Not really used yet, has some function to select composers with good amount of long songs.

networks.py:
Contains a BaseNet implementation from which all other networks inherit, this takes care of things like training,
validating and saves the results.
Also slightly adjusted versions of the Pytorch ResNet and DenseNet implementations can be found here.

lstm.py:
A pytorch implementation of the LSTM part of the Acoustic scene classification paper.

simple_cnn.py:
An implementation of the CNN part of the Acoustic scene classification paper.

stupid_overwrites.py:
Some changes to make the standard DenseNet work with 1 channel data <--


Interesting papers:

Acoustic Scene Classification Using Parallel Combination of LSTM and CNN:
https://pdfs.semanticscholar.org/4e7d/ad845bd9e1d399bf729724442cb7404549d1.pdf?_ga=2.233794461.2002662107.1557307441-282061435.1556630897

Multivariate LSTM-FCNs for Time Series Classification:
https://paperswithcode.com/paper/multivariate-lstm-fcns-for-time-series