# DeepBeats
_DeepBeats_ is a deep LSTM trained on hundreds of MIDI files with a custom encoding scheme. Version 1.0 was made in 24 hours at the HackMIT hackathon. Since then, we have been updating and improving the project. 

# Training Data
https://mega.nz/#!ZxgAAIZB!oMYIyy7iLYtnpnwRsKOuVRttOVrAHdQ2-DqPil2s7Lc

## Encoding Scheme
We use the pretty_midi library to convert each MIDI file into list of notes. With those notes, we construct a binary matrix _N_ x128 where _N_ is the length of the song in sixteenth notes, conversion = (length_in_seconds / (15 / tempo)). We then segment the matrix with a `look_back` parameter for training the LSTM on sequential data. Our preprocess step can be run with 

`python3 preprocess.py <path_to_midi_files> <output_dir> -w <look_back_window>`

and outputs a corpus of data and associated labels for training per song. Additionally for model debugging, the corpus can be processed and loaded in the same step then piped directly to the trainer. The function `pipe_corpus` in `preprocess.py` accomplishes this.

## Training

We use the Keras library to train a three layer LSTM. The model accepts a vector of size 128 as input and outputs a vector of the same size with distinct probabilities for each note (sigmoid activation) indicating whether that note should be _on_ or not. We count a note as _on_ if its value is >= 1 standard deviation from the mean. More detail can be found in `train.py`. The training step can be run with

`python3 train.py <path_to_corpus> <look_back_window> -n <output_file_name> -e <epochs> -c <capacity>`

This outputs a `.hd5` model to `./models`. The model can then be run to generate a song with default tempo of 120 bpm

`python3 generate.py <path_to_model> <path_to_sample_midi_files> <length_in_seconds>`
