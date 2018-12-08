from pathlib import Path
import pickle
import librosa
import numpy as np
from python_speech_features import mfcc
from os.path import dirname, abspath
import sys

parent_dir = dirname(abspath(__file__))

def load_all_data():

    clean_path = parent_dir + '/data/clean_trainset_wav_16k/'
    noisy_path = parent_dir + '/data/noisy_trainset_wav_16k/'

    clean_audios = load_data(clean_path, how_many=0)
    noisy_audios = load_data(noisy_path, how_many=0)

    save_as_pickled_object(clean_audios, parent_dir + '/pickles/clean_audios.pkl')
    save_as_pickled_object(noisy_audios, parent_dir + '/pickles/noisy_audios.pkl')

    return clean_audios, noisy_audios

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    """
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def load_data(dir_path, how_many=0):
    """

    Args:
        dir_path: path to the directory with txt and audio files.
        how_many: Integer. Number of directories we want to iterate,
                  that contain the audio files and transcriptions.

    Returns:
        txts: The spoken texts extracted from the .txt files,
              which correspond to the .flac files in audios.
              Text version.
        audios: The .flac file paths corresponding to the
                sentences in txts. Spoken version.

    """
    dir_path = Path(dir_path)
    audio_list = [f for f in dir_path.glob('*.wav') if f.is_file()]

    print('Number of audio file paths:', len(audio_list))

    audios = []

    # for development we want to reduce the numbers of files we read in.
    if how_many == 0:
        how_many = len(audio_list)

    for i, audio in enumerate(audio_list[:how_many]):
        print('Audio#:', i+1)
        audios.append(audioToInputVector(audio))

    return audios

def audioToInputVector(audio_filename):
    """
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.

    Borrowed from Mozilla's Deep Speech and slightly modified.
    https://github.com/mozilla/DeepSpeech
    """

    audio, fs = librosa.load(audio_filename)

    # # Get mfcc coefficients
    features = mfcc(audio,
        samplerate=fs,
        winlen=0.032,
        winstep=0.01,
        numcep=128,
        nfilt=128,
        nfft=1024,
        lowfreq=125,
        highfreq=7500)

    '''
    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)

    features = np.concatenate((empty_context, features, empty_context))
    

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2 * numcontext + 1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])
    '''

    # Whiten inputs (TODO: Should we whiten?)
    # Copy the strided array so that we can write to it safely
    train_inputs = np.log(features)
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # Return results
    return train_inputs

load_all_data()




