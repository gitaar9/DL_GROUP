import os
from enum import Enum
from random import randrange

import numpy as np
import torch
from torch.utils.data import Dataset


class Mode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class MidiClassicMusic(Dataset):
    """
    MidiClassicMusic dataset for PyTorch
    """
    def __init__(self, folder_path="./data/midi_files_npy_8_40", composers=['Brahms'], mode=Mode.TRAIN, slices=7, add_noise=True, unsqueeze=True, cv_cycle=0):
        """
        :param folder_path: The folder in which the .npy files can be found
        :param composers: A list of names of composers that should be loaded
        :param mode: How this dataset is used train/validation/test
        :param slices: The amount of slices that should be concatenated for one data sample
        :param add_noise: When True some extra noise is added to the data samples
        :param unsqueeze: When using a CNN the input should be in the form of [height, width, channels],
        when unsqueeze is True this dataset will output samples in that format otherwise just [height, width]
        :param cv_cycle: Which parts are currently train and which are validation
        """
        self.slices = slices
        self.mode = mode
        self.composers = composers
        self.amount_of_composers = len(composers)
        self.data_array, self.song_idxs_array = self.load_npy_files(folder_path, cv_cycle)
        self.add_noise = add_noise
        self.unsqueeze = unsqueeze

        # if self.mode != Mode.TRAIN:
        self.songs_to_simple_dataset_for_validations(step_size=None if self.mode != mode.TRAIN else 1)

    def __len__(self):
        # if self.mode == Mode.TRAIN:
        #     return 4000
        # else:
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

        torch_data = torch.from_numpy(song_image).float()
        return torch_data, index

    def get_validation_item(self, index):
        torch_data = torch.from_numpy(self.data_array[index]).float()
        return torch_data, self.validation_labels[index]

    def __getitem__(self, index):
        # if self.mode == Mode.TRAIN:
        #     torch_data, label = self.get_train_item(index)
        # else:
        torch_data, label = self.get_validation_item(index)

        if self.unsqueeze:
            torch_data = torch_data.unsqueeze(0)

        return torch_data, label

    def load_npy_files(self, folder_path, cv_cycle):
        data_filenames = ["{}_data.npy".format(composer) for composer in self.composers]
        data_array = []
        song_idxs_array = []

        for data_filename in data_filenames:
            songs = np.load(os.path.join(folder_path, data_filename))
            song_idx_filename = "{}_song_idxs.npy".format(data_filename.split('_')[0])
            song_idxs = np.load(os.path.join(folder_path, song_idx_filename))

            # Remove songs smaller than the input of the network
            songs, song_idxs = self.remove_to_small_songs(songs, song_idxs)
            # Split data in to cross_validation data and test_data
            cv_songs, cv_song_idxs, test_songs, test_song_idxs = self.take_out_every_nth_song(songs, song_idxs, 10)

            if self.mode == Mode.TEST:
                data_array.append(test_songs)
                song_idxs_array.append(test_song_idxs)
            else:
                # Cycle the song array
                amount_of_cycled_slices = cv_song_idxs[cv_cycle]
                slices_in_first_part = cv_songs[amount_of_cycled_slices:].shape[0]
                cv_songs = np.concatenate((cv_songs[amount_of_cycled_slices:], cv_songs[:amount_of_cycled_slices]), axis=0)

                # Cycle the idxs array
                first_part = cv_song_idxs[cv_cycle:] - amount_of_cycled_slices
                last_part = cv_song_idxs[:cv_cycle] + slices_in_first_part
                cv_song_idxs = np.concatenate((first_part, last_part))

                # Take every 4th for the validation
                train_songs, train_song_idxs, validation_songs, validation_song_idxs = self.take_out_every_nth_song(cv_songs, cv_song_idxs, 4)
                if self.mode == Mode.TRAIN:
                    data_array.append(train_songs)
                    song_idxs_array.append(train_song_idxs)
                else:
                    data_array.append(validation_songs)
                    song_idxs_array.append(validation_song_idxs)

        return data_array, song_idxs_array

    def songs_to_simple_dataset_for_validations(self, step_size=None):
        # For validation data we keep it a little bit simpler
        step_size = step_size or int(self.slices / 8)
        validation_data = []
        self.validation_labels = []
        for composer_idx, song_idxs in enumerate(self.song_idxs_array):
            for song_idx in range(len(song_idxs)):
                song_slices, length = self.get_song_slices(self.data_array[composer_idx], song_idxs, song_idx)
                slice_idx = 0
                while (slice_idx + self.slices) < length:
                    validation_data.append(np.concatenate(song_slices[slice_idx:slice_idx + self.slices], axis=1))
                    self.validation_labels.append(composer_idx)
                    slice_idx += step_size
        self.data_array = np.array(validation_data)

    def find_to_small_song(self, songs, song_idxs):
        for song_idx, song_start in enumerate(song_idxs):
            song_length_in_slices = song_idxs[song_idx + 1] - song_start if song_idx + 1 < len(song_idxs) else len(
                songs) - song_start
            if song_length_in_slices < self.slices:
                return song_idx
        return None

    @staticmethod
    def remove_song(song_idx, songs, song_idxs):
        """
        :param song_idx: The index of the song to be removed
        :param songs: Song slices array of one composer
        :param song_idxs: The start of every song in the songs array
        :return: songs and song_idxs but with the song_idxth song removed
        """
        if song_idx + 1 < len(song_idxs):  # We are not at the end of the array
            song_length = song_idxs[song_idx + 1] - song_idxs[song_idx]
        else:
            song_length = len(songs) - song_idxs[song_idx]
        songs = np.concatenate((songs[:song_idxs[song_idx]], songs[song_idxs[song_idx] + song_length:]), axis=0)
        song_idxs = np.concatenate((song_idxs[:song_idx], song_idxs[song_idx + 1:] - song_length))
        return songs, song_idxs

    def remove_to_small_songs(self, songs, song_idxs):
        song_to_be_removed = self.find_to_small_song(songs, song_idxs)
        while song_to_be_removed is not None:
            songs, song_idxs = self.remove_song(song_to_be_removed, songs, song_idxs)
            song_to_be_removed = self.find_to_small_song(songs, song_idxs)
        return songs, song_idxs

    def take_out_every_nth_song(self, songs, song_idxs, n):
        nth_songs = None
        nth_song_idxs = []
        for idx in range(len(song_idxs)-1, -1, -1):  # Walk through the array backwards
            if idx % n == 0:
                nth_song_idxs.append(0 if nth_songs is None else nth_songs.shape[0])
                nth_song = songs[song_idxs[idx]:song_idxs[idx+1]] if idx + 1 < len(song_idxs) else songs[song_idxs[idx]:len(songs)]
                nth_songs = nth_song if nth_songs is None else np.concatenate((nth_songs, nth_song))
                songs, song_idxs = self.remove_song(idx, songs, song_idxs)
        return songs, song_idxs, nth_songs, np.array(nth_song_idxs)
