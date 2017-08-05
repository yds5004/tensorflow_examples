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
    epochs = 100000000
    learning_rate = 0.1

    Data.process_data(filename=data_file, path=data_path)
    X, Y, idx2ch, ch2idx = Data.load_data(path=data_path)

    # X: (520571, 10)
    # Y: (520571, 10)
    trainset = Utils.rand_batch_gen(X, Y, batch_size=batch_size)

    # build the model
    num_classes = len(idx2ch)
    net = CharRNN(seqlen= X.shape[-1],
                  num_classes= num_classes,
                  num_layers= num_layers,
                  state_size= state_size,
                  epochs= epochs,
                  learning_rate= learning_rate,
                  batch_size= batch_size,
                  ckpt_path= ckpt_path,
                  model_name= model_name)

    # train on trainset
    net.train(trainset)