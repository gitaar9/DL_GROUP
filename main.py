import numpy as np
import os


def composer_with_at_least(folder_path, slices, songs, less_than_songs=999):
    data_filenames = [f for f in os.listdir(folder_path) if f.endswith("_data.npy")]
    qualified_composers = []
    total_slices = 0
    total_songs = 0
    for data_filename in data_filenames:
        song_data = np.load(os.path.join(folder_path, data_filename))
        song_idx_filename = "{}_song_idxs.npy".format(data_filename.split('_')[0])
        song_idxs = np.load(os.path.join(folder_path, song_idx_filename))

        qualified_songs = 0
        qualified_slices = 0
        for song in range(len(song_idxs)):
            song_end = song_idxs[song+1] if song < (len(song_idxs) - 1) else song_data.shape[0]
            song_length = song_end - song_idxs[song]
            if song_length >= slices:
                qualified_songs += 1
                qualified_slices += song_length

        if songs <= qualified_songs < less_than_songs:
            qualified_composers.append(data_filename.split('_')[0])
            total_slices += qualified_slices
            total_songs += qualified_songs
            print("{}: {} songs, {} slices".format(data_filename, qualified_songs, qualified_slices))
    return qualified_composers, total_slices, total_songs


if __name__ == '__main__':
    composers, total_slices, total_songs = composer_with_at_least('./data/midi_files_npy_8_40/', 40, 40)
    print(len(composers), total_slices, total_songs, composers)

# >10 <40 ['Scarlatti', 'Rachmaninov', 'Grieg', 'Buxehude', 'Liszt', 'Albéniz', 'Schumann', 'German', 'Skriabin', 'Tchaikovsky', 'Chaminade', 'Burgmuller', 'Paganini', 'Hummel', 'Czerny', 'Joplin', 'Debussy', 'Dvorak']
# >40 ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Beethoven', 'Haendel', 'Bach', 'Chopin']