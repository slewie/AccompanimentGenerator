from typing import List, Tuple



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