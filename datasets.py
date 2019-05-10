import os
from random import randrange

import numpy as np
import torch
from torch.utils.data import Dataset


class MidiClassicMusic(Dataset):
    """
    MidiClassicMusic dataset for PyTorch
    """
    def __init__(self, folder_path, composers, train=True, slices=7, add_noise=True, unsqueeze=True):
        self.slices = slices
        self.train = train
        self.composers = composers
        self.amount_of_composers = len(composers)
        self.data_array, self.song_idxs_array = self.load_npy_files(folder_path)
        self.add_noise = add_noise
        self.unsqueeze = unsqueeze

        # For validation data we keep it a little bit simpler
        if not self.train:
            validation_data = []
            self.validation_labels = []
            for composer_idx, song_idxs in enumerate(self.song_idxs_array):
                for song_idx in range(len(song_idxs)):
                    song_slices, length = self.get_song_slices(self.data_array[composer_idx], song_idxs, song_idx)
                    slice_idx = 0
                    while (slice_idx + self.slices) < length:
                        validation_data.append(np.concatenate(song_slices[slice_idx:slice_idx + self.slices], axis=1))
                        self.validation_labels.append(composer_idx)
                        slice_idx += 1
            self.data_array = np.array(validation_data)

    def __len__(self):
        if self.train:
            return 4000
        else:
            return len(self.validation_labels)

    @staticmethod
    def get_song_slices(composer_array, song_idxs, song_idx):
        song_start = song_idxs[song_idx]
        song_length_in_slices = song_idxs[song_idx+1] - song_start if song_idx+1 < len(song_idxs) else len(composer_array) - song_start
        return composer_array[song_start:song_start + song_length_in_slices], song_length_in_slices

    def get_train_item(self, index):
        index = index % self.amount_of_composers
        song_idxs = self.song_idxs_array[index]
        composer_data = self.data_array[index]

        amount_of_songs = len(song_idxs)
        song_length_in_slizes = 0
        while song_length_in_slizes <= self.slices:
            song_slices, song_length_in_slizes = self.get_song_slices(composer_data, song_idxs, randrange(amount_of_songs))

        start_of_part = randrange(song_length_in_slizes - self.slices)
        song_image = np.concatenate(song_slices[start_of_part:start_of_part + self.slices], axis=1)
        if self.add_noise:
            song_image += np.random.normal(loc=0, scale=0.01, size=song_image.shape)  # Add some noise

        torch_data = self.unsqueeze_tensor(torch.from_numpy(song_image)).float()
        return torch_data, index

    def get_validation_item(self, index):
        torch_data = self.unsqueeze_tensor(torch.from_numpy(self.data_array[index])).float()
        return torch_data, self.validation_labels[index]

    def unsqueeze_tensor(self, tensor):
        if self.unsqueeze:
            return tensor.unsqueeze(0)
        else:
            return tensor

    def __getitem__(self, index):
        return self.get_train_item(index) if self.train else self.get_validation_item(index)

    def load_npy_files(self, folder_path):
        data_filenames = ["{}_data.npy".format(composer) for composer in self.composers]
        # data_filenames = [f for f in os.listdir(folder_path) if f.endswith("_data.npy")]
        data_array = []
        song_idxs_array = []

        for data_filename in data_filenames:
            songs = np.load(os.path.join(folder_path, data_filename))
            song_idx_filename = "{}_song_idxs.npy".format(data_filename.split('_')[0])
            song_idxs = np.load(os.path.join(folder_path, song_idx_filename))

            # Split train/validation data 0.8/0.2
            amount_of_songs = len(song_idxs)
            first_validation_song = int(amount_of_songs * 0.8)
            if self.train:
                songs = songs[:song_idxs[first_validation_song]]
                song_idxs = song_idxs[:first_validation_song]
            else:
                songs = songs[song_idxs[first_validation_song]:]
                song_idxs = song_idxs[first_validation_song:]
                song_idxs -= song_idxs[0]

            data_array.append(songs)
            song_idxs_array.append(song_idxs)

        return data_array, song_idxs_array
