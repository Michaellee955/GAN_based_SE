import numpy as np
import tensorflow as tf

import librosa
import pickle
from collections import Counter
# import IPython.display as ipd
import matplotlib.pyplot as plt
# %matplotlib inline

import SpeechRecognizer
import sr_data_utils
import sr_model_utils


# Load data and process audios
# the function gets the path to the directory where the files are stored.
# it iterates the dir and subdirs and then processes the audios
# using the 'audioToInputVector' function
# and loads the corresponding text to each audio file.
from os.path import dirname, abspath
parent_dir = dirname(abspath(__file__))


def load_data(parent_dir):
    libri_path = parent_dir + '/data/LibriSpeech/dev-clean'
    txts, audios, audio_paths = sr_data_utils.load_data(libri_path, how_many=0)
    # ## to avoid having to process the texts and audios each time we can save them for later use.
    sr_data_utils.write_pkl(parent_dir + '/pickles/dev_txts.pkl', txts)
    sr_data_utils.write_pkl(parent_dir + '/pickles/dev_audio_paths.pkl', audio_paths)
    sr_data_utils.save_as_pickled_object(audios, parent_dir + '/pickles/dev_audios.pkl')
    return txts,audios,audio_paths


def reload_data(parent_dir):
    # and load them.
    txts = sr_data_utils.load_pkl(parent_dir+'/pickles/dev_txts.pkl')
    audio_paths = sr_data_utils.load_pkl(parent_dir+'/pickles/dev_audio_paths.pkl')
    audios = sr_data_utils.load_pkl(parent_dir+'/pickles/dev_audios.pkl')
    return txts,audios,audio_paths

def load(reload=False):
    if reload==True:
        txts,audios,audio_paths = reload_data(parent_dir)
    else:
        txts, audios, audio_paths = load_data(parent_dir)
    return txts,audios,audio_paths


def train(parent_dir,char2ind,ind2char,audios,txts_converted,restore=False):
    sr_model_utils.reset_graph()
    sr = SpeechRecognizer.SpeechRecognizer(char2ind,
                                           ind2char,
                                           parent_dir+'/models/models_1000points_0/my_model',
                                           num_layers_encoder=2,
                                           num_layers_decoder=2,
                                           rnn_size_encoder=450,
                                           rnn_size_decoder=450,
                                           embedding_dim=10,
                                           batch_size=40,
                                           epochs=500,
                                           use_cyclic_lr=True,
                                           learning_rate=0.00001,
                                           max_lr=0.00003,
                                           learning_rate_decay_steps=700)

    sr.build_graph()
    if restore==False:
        sr.train(audios[:1000],
                 txts_converted[:1000])
    else:
        sr.train(audios[:1000],
                 txts_converted[:1000],
                 restore_path=parent_dir+'/models/models_1000points_0/my_model')

# we are terribly overfitting here. therefore it won't generalize well.
# Note: hidden training process. Loss decreased to about 0.4


def test(char2ind,ind2char,parent_dir,audios):
    sr_model_utils.reset_graph()
    sr = SpeechRecognizer.SpeechRecognizer(char2ind,
                                           ind2char,
                                           parent_dir+'/models/models_1000points_0/my_model',
                                           num_layers_encoder=2,
                                           num_layers_decoder=2,
                                           rnn_size_encoder=450,
                                           rnn_size_decoder=450,
                                           mode='INFER',
                                           embedding_dim=10,
                                           batch_size=1,
                                           beam_width=5)

    sr.build_graph()
    preds = sr.infer(audios[0:500:20],
                    parent_dir+'/models/models_1000points_0/my_model')
    return preds

def main():
    txts,audios,audio_paths=load(reload=False)


    # the 'process_txts' function calls 'split_txts', 'create_lookup_dicts' and 'convert_txt_to_inds' internally.
    specials = ['<EOS>', '<SOS>', '<PAD>']
    txts_splitted, unique_chars, char2ind, ind2char, txts_converted = sr_data_utils.process_txts(txts, specials)

    # write lookup dicts to .pkl for later use.
    sr_data_utils.write_pkl(parent_dir+'/pickles/sr_word2ind.pkl', char2ind)
    sr_data_utils.write_pkl(parent_dir+'/pickles/sr_ind2word.pkl', ind2char)

    # Sort texts by text length or audio length from shortest to longest.
    # To keep everything in order we also sort the rest of the data.
    txts, audios, audio_paths, txts_splitted, txts_converted = sr_data_utils.sort_by_length(audios,
                                                                                            txts,
                                                                                            audio_paths,
                                                                                            txts_splitted,
                                                                                            txts_converted,
                                                                                            by_text_length=False)
    train(parent_dir,char2ind,ind2char,audios,txts_converted,restore=False)

    preds=test(char2ind,ind2char,parent_dir,audios)

    # preds2txt converts the predictions to text and removes <EOS> and <SOS> tags.
    preds_converted = sr_data_utils.preds2txt(preds, ind2char, beam=True)

    # prints the created texts side by side with the actual texts and
    # prints out an accuracy score of how good the prediction was.
    # if the created text is shorter than the actual one we
    # penalize by subtracting 0.3 (pretty hard!)
    # the accuracy clearly suffers, as the sequences get longer.
    sr_data_utils.print_samples(preds_converted, txts_splitted[0:500:20])

main()




