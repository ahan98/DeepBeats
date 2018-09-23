# DeepBeats
A deep learning model that uses MIDI files to generate music of a similar genre.

## Introduction
_DeepBeats_ is a deep LSTM trained on thousands of MIDI files with a custom encoding scheme. It was made in 24 hours at the HackMIT hackathon.

## Encoding Scheme
We use the music21 and pretty_midi libraries to convert each MIDI file into a binary matrix Nx128 where N is the length of the song in sixteenth notes. We then segment the matrix with a `look_back` parameter for training the LSTM on sequential data. Our preprocess step can be run with 

`python3 preprocess.py <path_to_midi_files> <output_dir> -w <look_back_window>`

and outputs a corpus of data and associated labels for training per song. Additionally, the corpus can be processed and loaded in the same step then piped directly to the model for training. The function `pipe_corpus` in `preprocess.py` accomplishes this.

## Training

We use the Keras library with tensorflow backend to train a three layer LSTM. The model accepts a vector of size 128 as input and outputs a similar vector of 128 with distinct probabilities for each note (using a sigmoid activation). More detail can be found in `train.py`. The training step can be run with

`python3 train.py <path_to_processed_data> <look_back_window> -n <model_name> -e <epochs>`


