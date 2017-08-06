import math
import numpy as np
import tensorflow as tf
import gensim

class NNUtils(object):

    # 데이터 + padding
    def pad_Xsequences_train(sentences, maxlen):
        nullItem = []
        itemLen = len(sentences[0][0])
        for i in range(0, itemLen):
            nullItem.append(0.0)

        newVec = []
        len1 = len(sentences)
        for i in range(0, len1):
            vec = []
            len2 = len(sentences[i])
            for j in range(0, len2):
                vec.append(sentences[i][j])
                j += 1
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='f')

    # padding + 데이터
    def pad_Xsequences_test(sentences, maxlen):
        nullItem = []
        itemLen = len(sentences[0][0])
        for i in range(0, itemLen):
            nullItem.append(0.0)

        newVec = []
        len1 = len(sentences)
        for i in range(0, len1):
            vec = []
            len2 = len(sentences[i])
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            for j in range(0, len2):
                vec.append(sentences[i][j])
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='f')

    # 데이터 + padding
    def pad_Ysequences_train(label, maxlen):
        len1 = len(label)
        newVec = []
        nullItem = 0
        for i in range(0, len1):
            vec = []
            len2 = len(label[i])
            for j in range(0, len2):
                vec.append(label[i][j])
                j += 1
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='int32')

    # padding + 데이터
    def pad_Ysequences_test(label, maxlen):
        len1 = len(label)
        newVec = []
        nullItem = 0
        for i in range(0, len1):
            vec = []
            len2 = len(label[i])
            for j in range(0, maxlen - len2):
                vec.append(nullItem)
                j += 1
            for j in range(0, len2):
                vec.append(label[i][j])
                j += 1
            newVec.append(vec)
            i += 1

        return np.array(newVec, dtype='int32')

    def xavier_init(n_inputs):
        stddev = math.sqrt(1.0 / (n_inputs))
        return tf.truncated_normal_initializer(stddev=stddev)

    def he_init(n_inputs):
        stddev = math.sqrt(2.0 / n_inputs)
        return tf.truncated_normal_initializer(stddev=stddev)

    def getNextBatch(index, batchSize, inFilename, sequenceSize, w2vModel):
        count = 0
        sentences = []
        labels = []
        start = index * batchSize
        end = (index + 1) * batchSize

        f = open(inFilename, 'r')
        while True:
            line = f.readline().strip('\n')
            if not line: break

            if count < start:
                count += 1
                continue
            if count >= end: break

            sentence = ''
            label = ''
            if line.count('\t') == 1:
                sentence, label = line.split('\t')
            elif line.count('\t') == 2:
                sentence1, sentence2, label = line.split('\t')
                sentence = sentence1 + '\t' + sentence2

            umjuls = [w2vModel.wv[w] for w in sentence]
            sentences.append(umjuls)

            umjuls = [w for w in label]
            labels.append(umjuls)
            count += 1

        f.close()

        inputX = NNUtils.pad_Xsequences_test(sentences, sequenceSize)
        inputY = NNUtils.pad_Ysequences_test(labels, sequenceSize)

        return inputX, inputY

    def getW2VModel(sentences, vectorSize=100, windowSize=10, minCount=1, sg=1):
        model = gensim.models.Word2Vec(sentences, size=vectorSize, window=windowSize, min_count=minCount, workers=4, sg=sg)
        return model

    def loadW2VModel(fileName):
        model = gensim.models.Word2Vec.load(fileName)
        return model

    def saveW2VModel(model, fileName):
        model.save(fileName)

    def getW2VSentence(sentences, w2vModel):
        w2v_sentences = []
        for i in range(len(sentences)):
            umjuls = [w2vModel.wv[w] for w in sentences[i]]
            w2v_sentences.append(umjuls)

        return w2v_sentences

    def to_array_vector(data, seqLen, vectorSize):
        # 한 줄의 길이를 구함
        num_chars = len(data)
        # 한 줄의 길이를 sequence_length으로 나눔
        dataLen = num_chars//seqLen
        # create numpy arrays
        X = np.zeros([dataLen, seqLen, vectorSize])
        Y = np.zeros([dataLen, seqLen, vectorSize])
        # data_len까지 반복하며, id로 채움
        for i in range(0, dataLen):
            X[i] = np.array([ vector for vector in data[i*seqLen:(i+1)*seqLen]])
            if (i == dataLen - 1 and ((i+1)*seqLen) + 1 > num_chars):
                temp = []
                for j in range(seqLen-1):
                    temp.append(data[(i*seqLen) + j + 1])
                temp.append(data[((i+1)*seqLen - 1)])
                Y[i] = np.array(temp)
            else:
                Y[i] = np.array([ch for ch in data[(i * seqLen) + 1: ((i + 1) * seqLen) + 1]])
        # return ndarrays
        return X, Y
