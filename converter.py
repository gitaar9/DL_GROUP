import os

import numpy as np
from music21 import midi


class Converter:

    def __init__(self, base_path="./data/midi-classic-music/", lowest_octave=1, highest_octave=6, precision=4,
                 slice_size=100):
        """
        :param base_path: The path where the midi files can be found
        :param lowest_octave: Midi notes with octaves lower then this will be transposed to this octave
        :param highest_octave: Same as lowest_octave but the other way around
        :param precision: How precise the created npy file is for a precision of 1 all notes played in the same bar are put together
        :param slice_size: How many timesteps are saved together in one slice of a song
        """
        self.base_path = base_path
        self.lowest_octave = lowest_octave
        self.highest_octave = highest_octave
        self.precision = precision
        self.array_height = 12 * (self.highest_octave - self.lowest_octave + 1)
        self.slice_size = slice_size

    def fill_note_in_array(self, array, offset, duration, octave, pitch):
        offset = int(offset * self.precision)
        height = (octave - self.lowest_octave) * 12 + pitch
        while height < 0:
            height += 12
        while height >= self.array_height:
            height -= 12
        for time_slice in range(offset, offset + int(duration * self.precision)):
            array[height][time_slice] += 1 if time_slice == offset else 0.5  # Only 1 on the first

    def load_midi_file(self, folder_path, filename):
        mf = midi.MidiFile()
        mf.open(os.path.join(folder_path, filename), 'rb')
        mf.read()  # read in the midi file
        mf.close()
        return mf

    def stream_to_2d_array(self, stream):
        array_width = int(stream.highestTime + 0.99) * self.precision

        array = np.zeros([self.array_height, array_width])
        for note in stream.flat.notes:
            if note.isChord:
                for pitch in note.pitches:
                    self.fill_note_in_array(array, note.offset, note.duration.quarterLength, pitch.octave, pitch.pitchClass)
            else:
                self.fill_note_in_array(array, note.offset, note.duration.quarterLength, note.pitch.octave, note.pitch.pitchClass)
        return array

    def slice_song(self, array_2d):
        sliced_array = None
        song_length = array_2d.shape[1]
        for start_of_slice_idx in range(0, song_length, self.slice_size):
            end_of_slice_idx = start_of_slice_idx + self.slice_size
            if end_of_slice_idx > song_length:
                break
            song_slice = np.array([array_2d[:, start_of_slice_idx:end_of_slice_idx]])
            sliced_array = np.concatenate([sliced_array, song_slice], axis=0) if sliced_array is not None else song_slice

        return sliced_array

    def folder_to_3d_array(self, folder_name, write_to_data_dict=True):
        folder_path = os.path.join(self.base_path, folder_name)
        filenames = [f for f in os.listdir(folder_path) if f.endswith(".mid") or f.endswith(".MID")]
        print("Found {} midi files for {}".format(len(filenames), folder_name))

        song_idxs = []
        folder_array = None
        # Songs with 10+ slices, song with 20+ slices
        metrics = [0, 0]
        for filename in filenames:
            try:
                midi_file = self.load_midi_file(folder_path, filename)
            except midi.MidiException:
                print('Skipping {} due to midi exception'.format(filename))
                continue
            try:
                song_stream = midi.translate.midiFileToStream(midi_file)
            except IndexError:
                print('Skipping {} due to midi -> stream translation error'.format(filename))
                continue

            song_2d_array = self.stream_to_2d_array(song_stream)

            song_idxs.append(folder_array.shape[0] if folder_array is not None else 0)  # So we know here songs start

            sliced_song_array = self.slice_song(song_2d_array)
            if sliced_song_array is not None:
                metrics[0] = metrics[0] + 1 if sliced_song_array.shape[0] >= 10 else metrics[0]
                metrics[1] = metrics[1] + 1 if sliced_song_array.shape[0] >= 20 else metrics[1]
                folder_array = np.concatenate([folder_array, sliced_song_array]) if folder_array is not None else sliced_song_array

        # Save some information like songs with 10+ slices, 20+ slices and total amount of slices
        if write_to_data_dict:
            with open('data/data_dict.csv', 'a+') as f:
                f.write("{}, {}, {}, {}\n".format(folder_name, folder_array.shape[0], metrics[0], metrics[1]))

        return folder_array, np.array(song_idxs)


if __name__ == '__main__':
    precision = 8
    slice_size = 40
    midi_directory = "./data/midi-classic-music/"
    npy_save_directory = "./data/midi_files_npy_{}_{}/".format(precision, slice_size)

    folders = [o for o in os.listdir(midi_directory) if os.path.isdir(os.path.join(midi_directory, o))]
    con = Converter(base_path=midi_directory, precision=precision, slice_size=slice_size)
    for folder_name in folders:
        try:
            np.load(os.path.join(npy_save_directory, "{}_song_idxs.npy".format(folder_name)))
            print("npy file already exists for {}".format(folder_name))
            continue
        except FileNotFoundError:
            pass

        # parse the midi files to a nparray
        array_3d, song_idxs = con.folder_to_3d_array(folder_name)

        # save the parsed midi as a .npy file
        folder_path = os.path.join(midi_directory, folder_name)
        np.save(os.path.join(npy_save_directory, "{}_data".format(folder_name)), array_3d)
        np.save(os.path.join(npy_save_directory, "{}_song_idxs".format(folder_name)), song_idxs)
        print("3D Array shape: {}\n{} song indexes: {}".format(array_3d.shape, len(song_idxs), song_idxs))
