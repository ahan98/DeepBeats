import sys
import os
import numpy as np
from music21 import *
from math import ceil
import pretty_midi

global FEATURE_SIZE
FEATURE_SIZE = 128

"""

PUBLIC METHODS

"""

''' convert a midi_file to a matrix'''
def midi_to_matrix(file, min_size=1):
	# make note stream and then create into matrix
	midi_data = pretty_midi.PrettyMIDI(file)
	notes = []
	for instr in midi_data.instruments:
		if instr.is_drum:
			notes = instr.notes
			break

	if not len(notes):
		return []

	d = 15.1 / midi_data.estimate_tempo()
	length = ceil(midi_data.get_end_time() / d)

	if length < min_size:
		return []

	M = np.zeros((length + 1, FEATURE_SIZE), dtype=np.bool)

	for _note in notes:
		dur = ceil((_note.end - _note.start) / d)
		start_time = ceil(_note.start / d) 
		midi_key_num = int(_note.pitch)

		# print(midi_key_num, start_time, (_note.end - _note.start))

		for i in range(dur):
			M[start_time + i][midi_key_num] = 1

	# for vec in M:
	# 	print(np.nonzero(vec))
		
	return M

''' write a matrix to a midi file'''
def matrix_to_midi(matrix, file_name='output', output_path='./'):

	song = _matrix_to_stream(matrix)

	# song.show('text')

	# write to a midi file
	midi_file = midi.translate.streamToMidiFile(song)

	if output_path[-1] != '/':
		output_path += '/'

	midi_file.open(output_path + file_name + '.mid', 'wb')
	midi_file.write()
	midi_file.close()

	print('[Success][Midi Process] Output midi file for', file_name, 'to', output)

''' extract percussion and write it to a separate midi file'''
def extract_percussion(file):
	mat = midi_to_matrix(file, 1)
	if not len(mat):
		raise Exception("Song too short")
	else:
		name = os.path.splitext(os.path.basename(file))[0]
		matrix_to_midi(mat, name, './')

"""

PRIVATE METHODS

"""

def _matrix_to_stream(matrix):
	matrix = np.concatenate((matrix, np.zeros((1, FEATURE_SIZE))), axis=0)

	time_count = 0.0
	step_size = float(16)
	notes_on = {}
	song = stream.Stream()

	for frame in matrix:
		# for each note in the matrix at frame x
		for i in np.nonzero(frame)[0]:
			if not i in notes_on:
				notes_on[i] = 1
			else:
				notes_on[i] += 1

		# for each note that is
		for i in list(notes_on.keys()):
			# note ended
			if not frame[i]:
				new_note = note.Note()
				new_note.pitch.ps = float(i)
				new_note.quarterLength = notes_on[i] / step_size
				new_note.offset = (time_count - notes_on[i]) / step_size
				song.append(new_note)

				del notes_on[i]

		time_count += 1.0

	return song

if __name__ == '__main__':
	extract_percussion('./sample_tracks/bongos.mid')
	# print(np.shape(mat))

