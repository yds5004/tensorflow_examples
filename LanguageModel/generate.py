from LanguageModel.Data import Data
from LanguageModel.Utils import Utils
from LanguageModel.CharRNN import CharRNN

import sys

if __name__ == '__main__':
    data_path = 'resources/data/'
    data_file = 'resources/tensorflow.txt'
    ckpt_path = 'resources/model/'
    model_name = 'tf_lm'

    batch_size = 128
    num_layers = 2
    state_size = 128
    epochs = 1000
    learning_rate = 0.1

    X, Y, idx2ch, ch2idx = Data.load_data(path=data_path)

    # build the model
    num_classes = len(idx2ch)
    net = CharRNN(seqlen= 1,
                  num_classes = num_classes,
                  num_layers = num_layers,
                  state_size = state_size,
                  epochs = 0,
                  learning_rate = 0.1,
                  batch_size = 1,
                  ckpt_path = ckpt_path,
                  model_name = model_name
            )

    # ic = initial character
    ic = 'd'
    # gc = the number of characters to generate
    gc = 1000

    char_indices = net.generate_characters(num_chars = gc, init_char_idx = ch2idx[ic])
    msg = ''.join([ idx2ch[chidx] for chidx in char_indices ])
    print(msg)