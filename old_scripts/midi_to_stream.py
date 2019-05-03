import numpy as np
from music21 import *


def load_midi_file(path):
    mf = midi.MidiFile()
    mf.open(path, 'rb')

    # read in the midi file
    mf.read()
    mf.close()
    return mf


def lowest_highest_octave(stream):
    # octaves = [note.pitch.octave for note in stream.flat.notes]
    # return min(octaves), max(octaves)
    lowest = 10
    highest = 0

    for note in stream.flat.notes:
        if note.isChord:
            octaves = [p.octave for p in note.pitches]
            lowest = min(octaves) if min(octaves) < lowest else lowest
            highest = max(octaves) if max(octaves) > highest else highest
        else:
            octave = note.pitch.octave
            lowest = octave if octave < lowest else lowest
            highest = octave if octave > highest else highest

    return lowest, highest


def fill_note_in_array(array, offset, duration, octave, pitch):
    offset = int(offset * 4)
    height = (octave - 1) * 12 + pitch
    while height < 0:
        height += 12
    while height > 71:
        height -= 12
    for time_slice in range(offset, offset + int(duration * 4)):
        array[height][time_slice] = 1


def stream_to_2d_array(stream):
    array_width = int(stream.highestTime + 0.95) * 4

    array = np.zeros([72, array_width])
    for note in stream.flat.notes:
        if note.isChord:
            for pitch in note.pitches:
                fill_note_in_array(array, note.offset, note.duration.quarterLength, pitch.octave, pitch.pitchClass)
        else:
            fill_note_in_array(array, note.offset, note.duration.quarterLength, note.pitch.octave, note.pitch.pitchClass)
    return array


def print_array(array):
    with open('output', 'w') as f:
        for pitch in range(array.shape[0]-1, -1, -1):
            f.write("".join([str(int(n)) for n in array[pitch, :]]) + '\n')


def print_events(event_array):
    print("\n".join([str(event.offset)+" "+str(event) for event in event_array]))


# filename = 'data/midi-classic-music/Satie/Gymnopedie No.1.mid'
filename = 'data/midi-classic-music/Rachmaninov/srapco31.mid'
# filename = 'data/midi-classic-music/Chopin/Etude No.1.mid'
mf = load_midi_file(filename)
stream = midi.translate.midiFileToStream(mf)

array = stream_to_2d_array(stream)
