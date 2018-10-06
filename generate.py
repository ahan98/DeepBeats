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

def generate_beat(model, training_data, song_len_in_seconds):
	
	song_len = song_len_in_seconds // (15 / 120)

	if training_data[-1] != '/':
		training_data += '/'

	look_back = model.layers[0].input_shape[1]

	X,Y = [],[]
	while not X:
		file = np.random.choice(glob.glob(training_data + '*.mid'))
		X, Y = make_training_data(file, look_back=look_back, partition_mat=True, write_to_file=False)

	start = np.random.randint(0, len(X)-1)
	pattern = X[start]
	matrix = deepcopy(pattern)

	# generate song_len note frames
	for note_index in range(song_len):
		# predict next vector
		next_input = np.reshape(pattern, (1, look_back, FEATURE_SIZE))
		prediction = model.predict(next_input, verbose=0)[0]
		
		# take every note 1 standard deviation above the mean
		sd = np.std(prediction)
		m = np.mean(prediction)
		thresh = 1*sd + m
		feature = [1 if prediction[i] >= thresh else 0 for i in range(len(prediction))]
		feature = np.array(feature)
	
		# step forward
		pattern.append(feature)
		pattern = pattern[1::]
		
		# save
		matrix.append(feature)

	return matrix

''' 
run on command line with args: 

MANDATORY:
<model>
<sampling corpus>
<length>
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', help='name of the model to use', action='store')
	parser.add_argument('corpus', help='path to sample midi files', action='store')
	parser.add_argument('length', help='length of the song to output', action='store', type=int)
	args = parser.parse_args()
	
	model = load_model(args.model)
	matrix = generate_beat(model, args.corpus, args.length)

	if not os.path.exists(_OUT_DIR):
		os.mkdir(_OUT_DIR)

	matrix_to_midi(matrix, 'generated_song', _OUT_DIR)
