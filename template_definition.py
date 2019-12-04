import cv2
#-------------------------------------------------------------------------------
# Template Paths
#-------------------------------------------------------------------------------

clef_paths = {
    "treble": [
        "resources/template/clef/treble_1.jpg",
        "resources/template/clef/treble_2.jpg", 
        "resources/template/clef/treble_3.jpg"
    ],
    "bass": [
        "resources/template/clef/bass_1.jpg"
    ]
}

accidental_paths = {
    "sharp": [
        "resources/template/sharp-line.png",
        "resources/template/sharp-space.png"
    ],
    "flat": [
        "resources/template/flat-line.png",
        "resources/template/flat-space.png"
    ]
}

note_paths = {
    "quarter": [
        "resources/template/note/quarter.png",
        "resources/template/note/solid-note.png"
    ],
    "half": [
        "resources/template/note/half-space.png",
        "resources/template/note/half-note-line.png",
        "resources/template/note/half-line.png",
        "resources/template/note/half-note-space.png"
    ],
    "whole": [
        "resources/template/note/whole-space.png",
        "resources/template/note/whole-note-line.png",
        "resources/template/note/whole-line.png",
        "resources/template/note/whole-note-space.png"
    ]
}
rest_paths = {
    "eighth": ["resources/template/rest/eighth_rest.jpg"],
    "quarter": ["resources/template/rest/quarter_rest.jpg"],
    "half": ["resources/template/rest/half_rest_1.jpg",
            "resources/template/rest/half_rest_2.jpg"],
    "whole": ["resources/template/rest/whole_rest.jpg"]
}

flag_paths = ["resources/template/flag/eighth_flag_1.jpg",
                "resources/template/flag/eighth_flag_2.jpg",
                "resources/template/flag/eighth_flag_3.jpg",
                "resources/template/flag/eighth_flag_4.jpg",
                "resources/template/flag/eighth_flag_5.jpg",
                "resources/template/flag/eighth_flag_6.jpg"]

barline_paths = ["resources/template/barline/barline_1.jpg",
                 "resources/template/barline/barline_2.jpg",
                 "resources/template/barline/barline_3.jpg",
                 "resources/template/barline/barline_4.jpg"]



#-------------------------------------------------------------------------------
# Template Images
#-------------------------------------------------------------------------------

# Clefs
clef_imgs = {
    "treble": [cv2.imread(clef_file, 0) for clef_file in clef_paths["treble"]],
    "bass": [cv2.imread(clef_file, 0) for clef_file in clef_paths["bass"]]
}

# Time Signatures
time_imgs = {
    "common": [cv2.imread(time, 0) for time in ["resources/template/time/common.jpg"]],
    "44": [cv2.imread(time, 0) for time in ["resources/template/time/44.jpg"]],
    "34": [cv2.imread(time, 0) for time in ["resources/template/time/34.jpg"]],
    "24": [cv2.imread(time, 0) for time in ["resources/template/time/24.jpg"]],
    "68": [cv2.imread(time, 0) for time in ["resources/template/time/68.jpg"]]
}

# Accidentals
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in accidental_paths["sharp"]]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in accidental_paths["flat"]]

# Notes
quarter_note_imgs = [cv2.imread(quarter, 0) for quarter in note_paths["quarter"]]
half_note_imgs = [cv2.imread(half, 0) for half in note_paths["half"]]
whole_note_imgs = [cv2.imread(whole, 0) for whole in note_paths['whole']]

# Rests
eighth_rest_imgs = [cv2.imread(eighth, 0) for eighth in rest_paths["eighth"]]
quarter_rest_imgs = [cv2.imread(quarter, 0) for quarter in rest_paths["quarter"]]
half_rest_imgs = [cv2.imread(half, 0) for half in rest_paths["half"]]
whole_rest_imgs = [cv2.imread(whole, 0) for whole in rest_paths['whole']]

# Eighth Flag
eighth_flag_imgs = [cv2.imread(flag, 0) for flag in flag_paths]

# Bar line
bar_imgs = [cv2.imread(barline, 0) for barline in barline_paths]


#-------------------------------------------------------------------------------
# Template Thresholds
#-------------------------------------------------------------------------------

# Clefs
clef_lower, clef_upper, clef_thresh = 50, 150, 0.50

# Time
time_lower, time_upper, time_thresh = 50, 150, 0.65

# # Clefs
# clef_lower, clef_upper, clef_thresh = 50, 150, 0.88

# # Time
# time_lower, time_upper, time_thresh = 50, 150, 0.85

# Accidentals
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77

# Notes
quarter_note_lower, quarter_note_upper, quarter_note_thresh = 50, 150, 0.70
half_note_lower, half_note_upper, half_note_thresh = 50, 150, 0.70
whole_note_lower, whole_note_upper, whole_note_thresh = 50, 150, 0.7011

# Rests
eighth_rest_lower, eighth_rest_upper, eighth_rest_thresh = 50, 150, 0.75 # Before was 0.7
quarter_rest_lower, quarter_rest_upper, quarter_rest_thresh = 50, 150, 0.70
half_rest_lower, half_rest_upper, half_rest_thresh = 50, 150, 0.80
whole_rest_lower, whole_rest_upper, whole_rest_thresh = 50, 150, 0.80

# Eighth Flag
eighth_flag_lower, eighth_flag_upper, eighth_flag_thresh = 50, 150, 0.8

# Bar line
bar_lower, bar_upper, bar_thresh = 50, 150, 0.85


#-------------------------------------------------------------------------------
# Mapping Functions
#-------------------------------------------------------------------------------

pitch_to_MIDI = {
    "C8": 108,
    "B7": 107,
    "Bb7": 106,
    "A#7": 106,
    "A7": 105,
    "Ab7": 104,
    "G#7": 104,
    "G7": 103,
    "Gb7": 102,
    "F#7": 102,
    "F7": 101,
    "E7": 100,
    "Eb7": 99,
    "D#7": 99,
    "D7": 98,
    "Db7": 97,
    "C#7": 97,
    "C7": 96,
    "B6": 95,
    "Bb6": 94,
    "A#6": 94,
    "A6": 93,
    "Ab6": 92,
    "G#6": 92,
    "G6": 91,
    "Gb6": 90,
    "F#6": 90,
    "F6": 89,
    "E6": 88,
    "Eb6": 87,
    "D#6": 87,
    "D6": 86,
    "Db6": 85,
    "C#6": 85,
    "C6": 84,
    "B5": 83,
    "Bb5": 82,
    "A#5": 82,
    "A5": 81,
    "Ab5": 80,
    "G#5": 80,
    "G5": 79,
    "Gb5": 78,
    "F#5": 78,
    "F5": 77,
    "E5": 76,
    "Eb5": 75,
    "D#5": 75,
    "D5": 74,
    "Db5": 73,
    "C#5": 73,
    "C5": 72,
    "B4": 71,
    "Bb4": 70,
    "A#4": 70,
    "A4": 69,
    "Ab4": 68,
    "G#4": 68,
    "G4": 67,
    "Gb4": 66,
    "F#4": 66,
    "F4": 65,
    "E4": 64,
    "Eb4": 63,
    "D#4": 63,
    "D4": 62,
    "Db4": 61,
    "C#4": 61,
    "C4": 60,
    "B3": 59,
    "Bb3": 58,
    "A#3": 58,
    "A3": 57,
    "Ab3": 56,
    "G#3": 56,
    "G3": 55,
    "Gb3": 54,
    "F#3": 54,
    "F3": 53,
    "E3": 52,
    "Eb3": 51,
    "D#3": 51,
    "D3": 50,
    "Db3": 49,
    "C#3": 49,
    "C3": 48,
    "B2": 47,
    "Bb2": 46,
    "A#2": 46,
    "A2": 45,
    "Ab2": 44,
    "G#2": 44,
    "G2": 43,
    "Gb2": 42,
    "F#2": 42,
    "F2": 41,
    "E2": 40,
    "Eb2": 39,
    "D#2": 39,
    "D2": 38,
    "Db2": 37,
    "C#2": 37,
    "C2": 36,
    "B1": 35,
    "Bb1": 34,
    "A#1": 34,
    "A1": 33,
    "Ab1": 32,
    "G#1": 32,
    "G1": 31,
    "Gb1": 30,
    "F#1": 30,
    "F1": 29,
    "E1": 28,
    "Eb1": 27,
    "D#1": 27,
    "D1": 26,
    "Db1": 25,
    "C#1": 25,
    "C1": 24,
    "B0": 23,
    "Bb0": 22,
    "A#0": 22,
    "A0": 21
}

MIDI_to_pitch = {val:key for key, val in pitch_to_MIDI.items()}

key_signature_changes = {
    "sharp": ["", "F", "FC", "FCG", "FCGD", "FCGDA", "FCGDAE", "FCGDAEB"],
    "flat": ["", "B", "BE", "BEA", "BEAD", "BEADG", "BEADGC", "BEADGCF"]
}