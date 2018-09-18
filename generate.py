'''
Use a model to generate beeps and boops
'''
import os
import argparse
import glob
import numpy as np
from keras.models import load_model
from copy import deepcopy
from midi_processing import matrix_to_midi, FEATURE_SIZE
from preprocess import make_training_data

_OUT_DIR = './generated/'

def generate_drums(model, training_data, song_len):
	return _generate_song(model, training_data, song_len, [10])

def generate_melody(model, training_data, song_len):
	return _generate_song(model, training_data, song_len, [i for i in range(16) if i != 10])

def generate_full(model, training_data, song_len):
	return []

def _generate_song(model, training_data, song_len, channels):

	if training_data[-1] != '/':
		training_data += '/'

	look_back = model.layers[0].input_shape[1]

	X,Y = [],[]
	while not X:
		file = np.random.choice(glob.glob(training_data + '*.mid'))
		X, Y = make_training_data(file, look_back=look_back, partition_mat=True, write_to_file=False, channels=channels)

	start = np.random.randint(0, len(X)-1)
	pattern = X[start]
	matrix = deepcopy(pattern)

	# generate 500 notes
	for note_index in range(song_len):
		next_input = np.reshape(pattern, (1, look_back, FEATURE_SIZE))

		prediction = model.predict(next_input, verbose=0)[0]
		sd = np.std(prediction)
		m = np.mean(prediction)
		thresh = 1*sd + m

		feature = [1 if prediction[i] >= thresh else 0 for i in range(len(prediction))]
		feature = np.array(feature)

		pattern.append(feature)
		pattern = pattern[1::]

		matrix.append(feature)

	return matrix

''' 
run on command line with args: 

MANDATORY:
<model>
<sampling corpus>
<length>

OPTIONAL:
-m <melody> 
-c <composite> 
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', help='name of the model to use', action='store')
	parser.add_argument('corpus', help='corpus for a random sample', action='store')
	parser.add_argument('length', help='length of the song to output', action='store', type=int)
	parser.add_argument('-m', '--melody', help='generate melody', action='store_true', default=False)
	parser.add_argument('-c', '--composite', help='generate melody and percussion', action='store_true', default=False)
	args = parser.parse_args()

	model = load_model(args.model)
	matrix = []

	if not args.melody:
		matrix = generate_drums(model, args.corpus, args.length)
	elif args.composite:
		matrix = generate_full(model, args.corpus, args.length)
	else:
		matrix = generate_melody(model, args.corpus, args.length)

	if not os.path.exists(_OUT_DIR):
		os.mkdir(_OUT_DIR)

	matrix_to_midi(matrix, 'generated_song', _OUT_DIR)
