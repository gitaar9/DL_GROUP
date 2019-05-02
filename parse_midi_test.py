from music21 import *


def min_max_events(event_array):
    notes = [note for note in event_array if note.isNoteOn()]
    pitches = [note.pitch for note in notes]
    return min(pitches), max(pitches)


def min_max_track(track):
    return min_max_events(track.events)


def print_events(event_array):
    print("\n".join([str(event) for event in event_array]))


def print_track(track):
    print_events(track.events)


def events_to_array(event_array, ticks_per_slice=120):
    lowest_note, highest_note = min_max_events(event_array)
    current_tick = 0
    notes_that_are_on = set([])
    moments = []
    for event in event_array:
        if event.isNoteOn():
            notes_that_are_on.add(event.pitch)
        elif event.isNoteOff():
            try:
                notes_that_are_on.remove(event.pitch)
            except KeyError:
                print("Turning of notes that wasnt on: {}".format(event.pitch))
        elif event.isDeltaTime():
            if event.time == 0:
                continue
            else:
                moments.append({'tick': current_tick, "currently on": notes_that_are_on.copy()})
                current_tick += event.time
    print(moments)


mf = midi.MidiFile()
filename = 'data/midi-classic-music/Satie/Gymnopedie No.1.mid'
# filename = 'data/midi-classic-music/Chopin/Etude No.1.mid'
mf.open(filename, 'rb')

# read in the midi file
mf.read()
print(mf.ticksPerQuarterNote)
print(mf.ticksPerSecond)
print(len(mf.tracks))

# show track 1

note_tracks = [t for t in mf.tracks if t.hasNotes()]
print(len(note_tracks))
first_30_events = note_tracks[0].events[:50]
print(min_max_events(first_30_events))
print_events(first_30_events)
events_to_array(first_30_events)
# print(mf)
mf.close()
