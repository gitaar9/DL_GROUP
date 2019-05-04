import os
from random import randrange

import numpy as np
import torch
from torch.utils.data import Dataset


class MidiClassicMusic(Dataset):
    """
    MidiClassicMusic dataset for PyTorch
    """
    def __init__(self, folder_path, train=True, slices=7):
        self.data_array, self.song_idxs_array = self.load_npy_files(folder_path, train)
        self.slices = slices

    def __len__(self):
        return len(self.song_idxs_array) * 1000

    def __getitem__(self, index):
        index = index % 4
        song_idxs = self.song_idxs_array[index]
        song_data = self.data_array[index]

        amount_of_songs = len(song_idxs)
        song_length = 0
        while song_length <= self.slices:
            chosen_song = randrange(amount_of_songs)
            song_length = (song_idxs[chosen_song+1] if (chosen_song + 1) < amount_of_songs else len(song_data)) - song_idxs[chosen_song]

        start_of_part_of_song = song_idxs[chosen_song] + randrange(song_length - self.slices)
        song_image = np.concatenate(song_data[start_of_part_of_song:start_of_part_of_song + self.slices], axis=1)
        torch_data = torch.from_numpy(song_image).unsqueeze(0).float()
        return torch_data, index

    @staticmethod
    def load_npy_files(folder_path, train):
        data_filenames = [f for f in os.listdir(folder_path) if f.endswith("_data.npy")]
        data_array = []
        song_idxs_array = []

        for data_filename in data_filenames:
            songs = np.load(os.path.join(folder_path, data_filename))
            song_idx_filename = "{}_song_idxs.npy".format(data_filename.split('_')[0])
            song_idxs = np.load(os.path.join(folder_path, song_idx_filename))

            # Split train/validation data 0.8/0.2
            amount_of_songs = len(song_idxs)
            first_validation_song = int(amount_of_songs * 0.8)
            if train:
                songs = songs[:song_idxs[first_validation_song]]
                song_idxs = song_idxs[:first_validation_song]
            else:
                songs = songs[song_idxs[first_validation_song]:]
                song_idxs = song_idxs[first_validation_song:]
                song_idxs -= song_idxs[0]

            data_array.append(songs)
            song_idxs_array.append(song_idxs)

        return data_array, song_idxs_array
