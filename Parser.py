from mido import MidiFile
import music21 as m21


class Parser:
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10, 12]
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
            for i in self.MAJOR_SCALE:
                notes.append(starting_note + i)
        else:
            for i in self.MINOR_SCALE:
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