import numpy as np
import os


def composer_with_at_least(folder_path, slices, songs):
    data_filenames = [f for f in os.listdir(folder_path) if f.endswith("_data.npy")]
    qualified_composers = []

    for data_filename in data_filenames:
        song_data = np.load(os.path.join(folder_path, data_filename))
        print(song_data.shape)
        song_idx_filename = "{}_song_idxs.npy".format(data_filename.split('_')[0])
        song_idxs = np.load(os.path.join(folder_path, song_idx_filename))

        qualified_songs = 0
        for song in range(len(song_idxs)):
            if song < (len(song_idxs) - 1):
                qualified_songs += 1 if (song_idxs[song+1] - song_idxs[song]) >= slices else 0
            else:
                qualified_songs += 1 if (song_data.shape[0] - song_idxs[song]) >= slices else 0
        if qualified_songs >= songs:
            qualified_composers.append(data_filename.split('_')[0])
    return qualified_composers


if __name__ == '__main__':
    composers = composer_with_at_least('./data/midi_files_npy/', 16, 10)
    print(len(composers), composers)
