'''
Train that model
'''
from preprocess import load_corpus

_OUT_DIR = './models/'

def make_model(look_back):

	model = Sequential()
	model.add(LSTM(FEATURE_SIZE, return_sequences=True, input_shape=(look_back, FEATURE_SIZE)))
	model.add(Dropout(0.1))
	model.add(LSTM(FEATURE_SIZE * 2, return_sequences=True))
	model.add(Dropout(0.1))
	model.add(LSTM(FEATURE_SIZE))
	model.add(Dense(FEATURE_SIZE))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def train_model(path, name, look_back, epochs):
	if path[-1] != '/':
		path += '/'

	print('Constructing training data...')

	X, Y = load_corpus(path, look_back)

	X = np.array(X)
	Y = np.array(Y)

	print('Corpus constructed with shape:',np.shape(X),'\nStarting training...')

	model = make_model(look_back)
	model.fit(X, Y, epochs=epochs, batch_size=FEATURE_SIZE, verbose=1)

	if not os.path.exists(_OUT_DIR):
		os.mkdir(_OUT_DIR)

	model.save(_OUT_DIR + name + '.h5')

	print('Outputting model to',_OUT_DIR + name + '.h5')
	print('Beep Boop. Goodbye.')

''' 
run on command line with args: 

MANDATORY:
<training>
<lookback>

OPTIONAL:
-n <name>
-e <epochs>
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('training', help='path to the training data', action='store')
	parser.add_argument('lookback', help='sequence length', type=int, action='store')
	parser.add_argument('-n', '--name', help='name of the output file', type=str, default='model')
	parser.add_argument('-e', '--epochs', help='training epochs', type=int, default=120)
	args = parser.parse_args()

	train_model(args.training, args.name, args.lookback, args.epochs)
