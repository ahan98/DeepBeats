import os
import sys
import numpy as np
import argparse
import glob
from collections import deque
from midi_processing import midi_to_matrix, FEATURE_SIZE

_OUTPUT_PATH = 'processed_data'
_LOOK_BACK_SIZE = 20

'''
PUBLIC METHODS
'''

''' make a corpus from a directory and save it '''
def make_corpus(dir_, save_path='./', look_back=20, channels=[10]):
	if not os.path.exists(dir_):
		raise Exception('The path to training data \'' + dir_ + '\' does not exist')

	if dir_[-1] != '/':
		dir_ += '/'

	patterns = 0
	for file_name in glob.glob(dir_ + '*.mid'):
		make_training_data(file_name, save_path=save_path, look_back=look_back, partition_mat=True, write_to_file=True, channels=channels)
		patterns += 1

	print('[Success][Make Corpus] Corpus with', patterns, 'samples generated')

''' make a corpus from a directory of n midi files with a given look_back and return it, for quick testing only '''
def pipe_corpus(dir_, look_back=20, n=sys.maxsize, channels=[10]):
    if not os.path.exists(dir_):
        raise Exception('The path to training data \'' + dir_ + '\' does not exist')

    if dir_[-1] != '/':
        dir_ += '/'

    # initialize X to be an array of look_back arrays where look_back arrays are FEATURE_SIZE tall
    X, Y = np.zeros((0,look_back, FEATURE_SIZE), dtype=np.bool), np.zeros((0,FEATURE_SIZE), dtype=np.bool)
    patterns = 0

    samples = 0
    for file_name in glob.glob(dir_ + '*.mid'):
        x, y = make_training_data(file_name, look_back=look_back, partition_mat=True, write_to_file=False, channels=channels)

        if not len(x):
            continue

        patterns += len(x)
        X = np.concatenate((X, x), axis=0)
        Y = np.concatenate((Y, y), axis=0)

        samples += 1
        if samples >= n:
            break

    X = np.reshape(X, (patterns, look_back, FEATURE_SIZE))
    Y = np.reshape(Y, (patterns, FEATURE_SIZE))

    return X, Y

''' load a saved corpus from a directory with a given look_back '''
def load_corpus(dir_, look_back=20):
    if not os.path.exists(dir_):
        raise Exception('The path to training data \'' + dir_ + '\' does not exist')

    X, Y = np.zeros((0,look_back, FEATURE_SIZE), dtype=np.bool), np.zeros((0, FEATURE_SIZE), dtype=np.bool)
    patterns = 0
    for file_name in os.listdir(dir_):
        folder_path = dir_ + '/' + file_name
        if os.path.isdir(folder_path):
            try:
                x = np.load(folder_path + '/data.npy')
                y = np.load(folder_path + '/labels.npy')

                patterns += len(x)
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)
            except:
                print('[Error][Load Corpus] Could not load file at', folder_path)

    return np.reshape(X, (patterns, look_back, FEATURE_SIZE)), np.reshape(Y, (patterns, FEATURE_SIZE))

''' given a midi file, return or save the sequence and labels '''
def make_training_data(midi_file, save_path='./', look_back=_LOOK_BACK_SIZE, partition_mat=True, write_to_file=True, channels=[10]):
	# convert to a matrix
	M = midi_to_matrix(midi_file, look_back, channels)

	if not len(M):
		print('[Error][Preprocess] Song too short')
		return [], []

	name = os.path.splitext(os.path.basename(midi_file))[0]

	# make sure training data folder and data folder is ready
	out = save_path
	if out[-1] != '/':
		out += '/'

	if write_to_file:
		out += _OUTPUT_PATH
		if not os.path.exists(out):
			os.mkdir(out)

		# if folder already exists, create a new one
		out += '/' + name
		if os.path.exists(out):
			i = 1
			while os.path.exists(out + '_' + str(i)):
				i += 1
			out += '_' + str(i)
		out += '/'

		# make output folder
		os.mkdir(out)

	# generate labeled traning data or just save matrix for later use
	if partition_mat:
		X, Y = _make_tuples(M, look_back, out)

		if write_to_file:
			np.save(out + 'data', X)
			np.save(out + 'labels', Y)
			print('[Success][Preprocess] Output training data for', name)
		
		return X, Y
	else:
		if write_to_file:
			np.save(out + 'matrix', M)
			print('[Success][Preprocess] Output matrix for', name)

		return M, []

'''
PRIVATE METHODS
'''	

''' generate sequences and labels '''
def _make_tuples(matrix, look_back, out):
	if look_back >= len(matrix):
		print('[Error][Preprocess] look back of', look_back, 'too large for song of length', len(matrix))
		return [], []

	X = []
	Y = []

	current = deque()
	next_row = None

	for i in range(look_back):
		current.append(matrix[i])
		next_row = matrix[i+1]

	X.append(list(current))
	Y.append(next_row)

	for i in range(look_back, len(matrix) - 1):
		current.popleft()
		current.append(matrix[i])
		next_row = matrix[i+1]

		X.append(list(current))
		Y.append(next_row)

	return X, Y

''' 
run on command line with args: 

MANDATORY:
<midi_file> 

OPTIONAL:
-o <output> 
-w <look_back> 
-s <save as matrix flag> 
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('files', help='path to the target midi file', action='store')
	parser.add_argument('out', help='output path', default='./', action='store')
	parser.add_argument('-w', '--window', help='look back size for generating training data', type=int, default=_LOOK_BACK_SIZE)
	args = parser.parse_args()

	make_corpus(args.files, args.out, args.window)
