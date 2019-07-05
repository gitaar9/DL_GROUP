# Composer prediction based on midi files

In this project we try to classify the author of a musical piece using [MIDI](https://en.wikipedia.org/wiki/MIDI) files.
We explore different configurations mixing CNNs and LSTMs.

## How to run it

```bash
git clone https://github.com/gitaar9/DL_GROUP.git
cd DL_GROUP

# Create virtual environment in repo directory
python3 -m venv venv

# Source it and install the requirements
source venv/bin/activate
pip install -r requirements.txt

# Unzip the dataset into <repo_dir>/data
unzip data/midi_files_npy_8_40.zip -d data/

# Run the appropriate file with the desired hyper parameters, for example:
python cnn-lstm.py --num_lstm_layers 1 --lstm_input_size 512 --lstm_hidden_size 1024 --chunk_size 160 --chunk_stride 80
```

## How to run on peregrine
1. create a venv folder in root directory ./
2. clone the DL_GROUP in root direcrtory or atleast at the same level as the venv(this is important for the jobscripts to activate the venv)
3. unzip the datafile you want to use
4. run sbatch yourjobscriopt

### Functions

**converter.py:**
This file reads in the midi files and stores them in npy files that can be easily read in while training a network.
There are several options of how to do this: precision, lowest/highest octave.
At the moment a song is sliced in parts of 100 timesteps wide, this should be refactored.

**datasets.py:**
Implements the classical midi dataset dataset, when training it will provide songs from all composers with equal chance.
This is important because otherwise the network would for instance see much more Mozart than Liszt.
While validating the dataset works more intuitively.

**main.py:**
Not really used yet, has some function to select composers with good amount of long songs.

**networks.py:**
Contains a BaseNet implementation from which all other networks inherit, this takes care of things like training,
validating and saves the results.
Also slightly adjusted versions of the Pytorch ResNet and DenseNet implementations can be found here.

**lstm.py:**
A pytorch implementation of the LSTM part of the Acoustic scene classification paper.

**cnn-lstm.py:**
Our implementation of a serial CNN-LSTM network

**parallel_cnn_lstm.py:**
Our implementation of a parallel CNN-LSTM network

**lstm_cnn_model.py:**
Our implementation of a serial LSTM-CNN network

**simple_cnn.py:**
An implementation of the CNN part of the Acoustic scene classification paper.

**simple_dense_cnn.py:**
Adapted the simple_cnn.py to work with densely connected blocks

**stupid_overwrites.py:**
Some changes to make the standard DenseNet work with 1 channel data <-- can be done simpler

**plotter.py:**
A python script to plot the results, use --help option for arguments


