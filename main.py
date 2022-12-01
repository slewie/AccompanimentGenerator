import sys
from mido import MidiFile, MidiTrack, Message
from typing import List
from genetic_algorithm import GeneticAlgorithm
from Parser import Parser


class Accompaniment:
    @staticmethod
    def create_track(chords: List[List[int]], bpm: int):
        global mid
        track = MidiTrack()
        track.append(mid.tracks[1][0])
        track.append(mid.tracks[1][1])
        for chord in chords:
            for note in chord:
                track.append(Message('note_on', note=note, velocity=40, time=0))
            track.append(Message('note_off', note=chord[0], velocity=0, time=bpm))
            track.append(Message('note_off', note=chord[1], velocity=0, time=0))
            track.append(Message('note_off', note=chord[2], velocity=0, time=0))
        mid.tracks.append(track)


if __name__ == '__main__':
    input_file = sys.argv[1]
    parser = Parser(input_file)
    parser.parse()
    ga = GeneticAlgorithm(30, parser.notes_allowed, int(parser.number_of_quarters), parser.notes_per_quarter, parser.chords,
        generations=300)
    pop, pop_fit = ga.evolution()
    mid = MidiFile(input_file)
    Accompaniment().create_track(pop[pop_fit.index(max(pop_fit))], parser.ticks_per_beat)
    output_file = "output-{}{}.mid".format(parser.tonic, "m" if parser.scale == "minor" else "")
    mid.save(output_file)
