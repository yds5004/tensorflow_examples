SMS_FILENAME = 'data/sms/sms.txt'
MADURAI_FILENAME = 'data/madurai/sample.txt'

MADURAI_PATH = 'data/madurai/'
SMS_PATH = 'data/sms/'


import csv
import numpy as np
import pickle as pkl

class Data(object):

    def read_lines_sms(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            return [ row[-1] for row in list(reader) ]

    def read_lines(filename):
        with open(filename) as f:
            return f.read().split('\n')

    def index_(lines):
        # 음절 단위로 unique한 음절을 list로 반환
        vocab = list(sorted(set('\n'.join(lines))))
        # 음절이 key, Id가 value인 dictionary를 반환
        ch2idx = { k:v for v,k in enumerate(vocab) }

        return vocab, ch2idx

    def to_array(lines, seqlen, ch2idx):
        # Document를 한 줄로 만듬
        raw_data = '\n'.join(lines)
        # 한 줄의 길이를 구함
        num_chars = len(raw_data)
        # 한 줄의 길이를 seqlen으로 나눔
        data_len = num_chars//seqlen
        # create numpy arrays
        X = np.zeros([data_len, seqlen])
        Y = np.zeros([data_len, seqlen])
        # data_len까지 반복하며, id로 채움
        for i in range(0, data_len):
            X[i] = np.array([ ch2idx[ch] for ch in raw_data[i*seqlen:(i+1)*seqlen] ])
            Y[i] = np.array([ ch2idx[ch] for ch in raw_data[(i*seqlen) + 1 : ((i+1)*seqlen) + 1] ])
        # return ndarrays
        return X, Y

    def process_data(path, filename, seqlen=10):
        # line 단위로 array를 구함
        lines = Data.read_lines(filename)
        # idx2ch: 음절 단위로 unique한 음절을 list로 반환
        # ch2idx: 음절이 key, Id가 value인 dictionary를 반환
        idx2ch, ch2idx = Data.index_(lines)
        # X: 음절에 대한 array list
        # Y: 음절의 id에 대한 array list
        X, Y = Data.to_array(lines, seqlen, ch2idx)
        # X와 Y array list에 대한 파일 저장
        np.save(path+ 'idx_x.npy', X)
        np.save(path+ 'idx_y.npy', Y)

        # idx2ch, ch2idx에 대한 사전 (dictionary) 저장
        with open(path+ 'metadata.pkl', 'wb') as f:
            pkl.dump( {'idx2ch' : idx2ch, 'ch2idx' : ch2idx }, f )

    def load_data(path):
        # read data control dictionaries
        with open(path + 'metadata.pkl', 'rb') as f:
            metadata = pkl.load(f)

        # read numpy arrays
        X = np.load(path + 'idx_x.npy')
        Y = np.load(path + 'idx_y.npy')
        return X, Y, metadata['idx2ch'], metadata['ch2idx']


if __name__ == '__main__':
    Data.process_data(path = MADURAI_PATH,filename = MADURAI_FILENAME)
