import random
import sys
from mido import MidiFile, MidiTrack, Message
import music21 as m21
import numpy as np
from typing import List, Tuple

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10, 12]


class Parser:
    """
    The class parses midi file and takes all the necessary information from it
    """

    def __init__(self, file: str):
        """
        :param file: path to midi file
        mid: opened midi file
        notes_per_quarter: list of notes per quarter
        notes_allowed: list of notes allowed in the key
        chords: list of chords allowed in the key
        ticks_per_beat: number of ticks per beat
        size_of_sixteenth: size of quarter in ticks
        number_of_quarters: number of quarters in the song
        duration: duration of the song in ticks
        """
        self.file = file
        self.mid = MidiFile(self.file)
        self.notes_per_quarter = []
        self.notes_allowed = []
        self.chords = []
        self.tonic = None
        self.octave = None
        self.scale = None
        self.ticks_per_beat = None
        self.size_of_sixteenth = None
        self.number_of_quarters = None
        self.duration = None

    def parse(self):
        """
        The method calls all methods to parse midi file
        """
        self.parse_key()
        self.parse_allowed_notes()
        self.parse_allowed_chords()
        self.parse_time()
        self.parse_note_per_quarter()

    def parse_key(self):
        """
        Using music21 library to parse midi file
        """
        score = m21.converter.parse(self.file)
        key = score.analyze('key')
        self.tonic = key.tonic
        self.octave = key.tonic.implicitOctave
        self.scale = key.mode

    @staticmethod
    def from_note_to_int(note: m21.pitch.Pitch, octave: int) -> int:
        """
        Get note number from note name and octave
        """
        return note.pitchClass + (octave * 12)

    def parse_allowed_notes(self):
        """
        Get notes allowed in the key with octave shift
        """
        notes = []
        octave_shift = -1
        starting_note = Parser.from_note_to_int(self.tonic, self.octave + octave_shift)
        if self.scale == 'major':
            for i in MAJOR_SCALE:
                notes.append(starting_note + i)
        else:
            for i in MINOR_SCALE:
                notes.append(starting_note + i)
        self.notes_allowed = notes

    def parse_allowed_chords(self):
        """
        Get chords allowed in the key
        """
        for note in self.notes_allowed:
            self.chords.append([note, note + 4, note + 7])
            self.chords.append([note, note + 3, note + 7])
            self.chords.append([note, note + 3, note + 6])

    def parse_time(self):
        """
        Parse time values
        """
        self.ticks_per_beat = self.mid.ticks_per_beat
        self.duration = 0
        self.size_of_sixteenth = self.ticks_per_beat // 4

        for track in self.mid.tracks:
            duration = 0
            for msg in track:
                if not msg.is_meta:
                    duration += msg.time
            self.duration = max(duration, self.duration)
        self.number_of_quarters = self.duration // self.ticks_per_beat + 1 if self.duration % self.ticks_per_beat != 0 \
            else self.duration // self.ticks_per_beat

    def parse_note_per_quarter(self):
        """
        Parse notes per tick
        """
        current_time = 0
        number_of_sixteenth = self.duration // self.size_of_sixteenth + 1 if self.duration % self.size_of_sixteenth != 0 else self.duration // self.size_of_sixteenth
        self.notes_per_quarter = [[0 for _ in range(4)] for _ in range(number_of_sixteenth)]
        for msg in self.mid.tracks[1]:
            if not msg.is_meta:
                current_time += msg.time
                if msg.type == 'note_on':
                    self.notes_per_quarter[current_time // self.ticks_per_beat][
                        int((current_time % self.ticks_per_beat) / self.size_of_sixteenth)] = msg.note


class Chord:
    """
    The class methods calculate the fitness values for the chord
    """

    def __init__(self, chord: List[int]):
        self.chord = chord

    def notes_in_chord(self, notes_in_chord: List[int]) -> int:
        """
        The method checks whether the chord contains all the notes in the chord
        """
        result = 0
        for note1 in notes_in_chord:
            for note2 in self.chord:
                if note1 % 12 == note2 % 12:
                    result += 50
        return result

    def chord_notes_distance(self, notes_in_chord: List[int]) -> int:
        """
        The method calculates the distance between the notes in the chord and the notes in the melody
        """
        result = 0
        for note1 in notes_in_chord:
            for note2 in self.chord:
                result += self.distance_between_notes(note1, note2)
        number_nonzero_elements = 0
        for note in notes_in_chord:
            if note != 0:
                number_nonzero_elements += 1
        return result // max(1, number_nonzero_elements)

    @staticmethod
    def distance_between_notes(note1: int, note2: int) -> int:
        """
        The method calculates the distance between two notes
        """
        if note1 != 0:
            difference = abs(note1 - note2) % 12
            if difference in [0, 7]:
                return 70
            elif difference in [5]:
                return 40
            elif difference in [2, 10]:
                return 40
            elif difference in [3, 4, 8, 9]:
                return 30
            elif difference in [6]:
                return 10
            elif difference in [10, 11]:
                return -10
        return 0

    def get_inverse_chord(self) -> List[List[int]]:
        """
        The method returns the inversions of the chord(2 inversions to the left and 2 to the right)
        """
        chords = []
        notes_up = self.chord[:]
        notes_down = self.chord[:]
        for i in range(2):
            notes_up.append(min(notes_up.pop(0) + 12, 127))
            chords.append(notes_up[:])
            notes_down.insert(0, max(0, notes_down.pop(-1) - 12))
            chords.append(notes_down[:])
        return chords

    def notes_chord_octave(self, notes_in_chord: List[int]) -> int:
        """
        The method checks whether the notes in the chord are in the same octave
        """
        notes_octaves = [note // 12 for note in notes_in_chord]
        chord_octaves = [note // 12 for note in self.chord]
        penalty = 0
        for octave1 in notes_octaves:
            for octave2 in chord_octaves:
                if octave1 - octave2 != 1:
                    penalty += 100
        return -penalty


class GA:
    """
    Individual - list of chords, each chord is a list of three notes. Length of this chord are one quarter
    Population - list of individuals
    Crossover type - two-point crossover
    Mutation - Random Resetting Mutation
    Selection - Roulette Wheel Selection
    """
    CHORD_SIZE = 3

    def __init__(self, population_size: int, notes: List[int], number_of_quarters: int,
                 notes_per_quarter: List[List[int]], chords: List[List[int]], mutation_rate=0.1,
                 generations=1000):
        self.population_size = population_size
        self.notes = notes
        self.number_of_quarters = number_of_quarters
        self.notes_per_quarter = notes_per_quarter
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.chords = chords

    def get_individual(self) -> List[List[int]]:
        chords = []
        for _ in range(self.number_of_quarters):
            chords.append(random.choice(self.chords))
        return chords

    def get_population(self) -> List[List[List[int]]]:
        population = [self.get_individual() for _ in range(self.population_size)]
        return population

    @staticmethod
    def get_chord_fitness(chord: List[int], notes_in_quart: List[int]) -> int:
        chord_ = Chord(chord)
        fitness_score = chord_.notes_in_chord(notes_in_quart)
        fitness_score += chord_.chord_notes_distance(notes_in_quart)
        fitness_score += chord_.notes_chord_octave(notes_in_quart)
        return fitness_score

    def get_individual_fitness(self, individual: List[List[int]]) -> int:
        fitness_score_all = 0
        for i in range(len(individual)):
            fitness_score = self.get_chord_fitness(individual[i], self.notes_per_quarter[i])
            if fitness_score <= 50:
                inverse_chords = Chord(individual[i]).get_inverse_chord()
                best_chord = individual[i]
                best_fitness = fitness_score
                for chord in inverse_chords:
                    fitness_score = self.get_chord_fitness(chord, self.notes_per_quarter[i])
                    if fitness_score > best_fitness:
                        best_chord = chord
                        best_fitness = fitness_score
                individual[i] = best_chord
                for j in range(len(individual[i])):
                    individual[i][j] = best_chord[j]
                fitness_score = best_fitness
            fitness_score_all += fitness_score
        return fitness_score_all

    def get_population_fitness(self, population: List[List[List[int]]]) -> List[int]:
        population_fitness = []
        for individual in population:
            population_fitness.append(self.get_individual_fitness(individual))
        return population_fitness

    @staticmethod
    def roulette_wheel_select(population_fitness: List[int]) -> int:
        shift = min(population_fitness)
        if shift < 0:
            population_fitness = [x - (shift - 5) for x in population_fitness]
        sum_fitness = sum(population_fitness)
        probabilities = [fitness / sum_fitness for fitness in population_fitness]
        return np.random.choice(range(len(population_fitness)), size=1, p=probabilities)[0]

    @staticmethod
    def two_point_crossover(parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[
        List[List[int]], List[List[int]]]:
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(0, len(parent1) - 1)
        while crossover_point1 == crossover_point2:
            crossover_point2 = random.randint(0, len(parent1) - 1)
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        offspring1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[
                                                                                               crossover_point2:]
        offspring2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[
                                                                                               crossover_point2:]
        return offspring1, offspring2

    def crossover(self, population: List[List[List[int]]], population_fitness: List[int], size: int) \
            -> List[List[List[int]]]:
        offsprings = []
        for _ in range(size):
            idx1, idx2 = self.roulette_wheel_select(population_fitness), self.roulette_wheel_select(population_fitness)
            while idx1 == idx2:
                idx2 = self.roulette_wheel_select(population_fitness)
            parent1, parent2 = population[idx1], population[idx2]
            offspring1, offspring2 = self.two_point_crossover(parent1, parent2)
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        return offsprings

    def mutation(self, offsprings: List[List[List[int]]]) -> List[List[List[int]]]:
        for offspring in offsprings:
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_rate:
                    offspring[i] = self.get_individual()[i]
        return offsprings

    @staticmethod
    def replace_parents(population: List[List[List[int]]], population_fitness: List[int],
                        offsprings: List[List[List[int]]], offsprings_fitness: List[int], size: int) -> List[
        List[List[int]]]:
        sort_index = np.argsort(population_fitness)
        population_sorted = [[] for _ in range(len(population))]
        for i in sort_index:
            population_sorted[i] = population[i]
        sort_index = np.argsort(offsprings_fitness)
        offsprings_sorted = [[] for _ in range(len(offsprings))]
        for i in sort_index:
            offsprings_sorted[i] = offsprings[i]
        parents = population_sorted[size:]
        offsprings = offsprings_sorted[-size:]
        return [*parents, *offsprings]

    def evolution(self):
        population = self.get_population()
        population_fitness = self.get_population_fitness(population)
        for i in range(self.generations):
            offsprings = self.crossover(population, population_fitness, self.population_size // 2)
            offsprings = self.mutation(offsprings)

            offsprings_fitness = self.get_population_fitness(offsprings)
            population = self.replace_parents(population, population_fitness, offsprings, offsprings_fitness,
                                              int(self.population_size * 0.3))
            population_fitness = self.get_population_fitness(population)
        return population, population_fitness


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
    ga = GA(30, parser.notes_allowed, int(parser.number_of_quarters), parser.notes_per_quarter, parser.chords,
        generations=300)
    pop, pop_fit = ga.evolution()
    mid = MidiFile(input_file)
    Accompaniment().create_track(pop[pop_fit.index(max(pop_fit))], parser.ticks_per_beat)
    output_file = "output-{}{}.mid".format(parser.tonic, "m" if parser.scale == "minor" else "")
    mid.save(output_file)
