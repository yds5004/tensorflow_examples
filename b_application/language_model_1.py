import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from random import sample
import math

# char의 id 사용

def rand_batch_gen(x, y, batchSize=1):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batchSize)
        yield x[sample_idx], y[sample_idx]

def to_array(data, ch2Idx, seqLen):
    # 한 줄의 길이를 구함
    num_chars = len(data)
    # 한 줄의 길이를 sequence_length으로 나눔
    dataLen = num_chars//seqLen
    # create numpy arrays
    X = np.zeros([dataLen, seqLen])
    Y = np.zeros([dataLen, seqLen])
    # data_len까지 반복하며, id로 채움
    for i in range(0, dataLen):
        X[i] = np.array([ ch2Idx[ch] for ch in data[i*seqLen:(i+1)*seqLen]])
        if (i == dataLen - 1 and ((i+1)*seqLen) + 1 > num_chars):
            temp = []
            for j in range(seqLen-1):
                temp.append(ch2Idx[data[(i*seqLen) + j + 1]])
            temp.append(ch2Idx[data[((i+1)*seqLen - 1)]])
            Y[i] = np.array(temp)
        else:
            Y[i] = np.array([ch2Idx[ch] for ch in data[(i * seqLen) + 1: ((i + 1) * seqLen) + 1]])
    # return ndarrays
    return X, Y

def train(trainSet, learningRate, epochNum, batchSize, seqLen, numClasses, stateSize):
    # reset graph
    tf.reset_default_graph()

    # placeholders
    x_ = tf.placeholder(tf.int64, [None, seqLen], name='x')
    y_ = tf.placeholder(tf.int64, [None, seqLen], name='y')

    embs = tf.get_variable('emb', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embs, x_)

    # parameters for softmax layer
    W = tf.Variable(tf.truncated_normal([state_size, numClasses], stddev=math.sqrt(2.0 / (seqLen))))
    b = tf.Variable(tf.random_uniform([numClasses], -1.0, 1.0))

    # rnn cell
    cell = rnn.GRUCell(stateSize)
    init_state = cell.zero_state(batchSize, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)
    rnn_outputs = tf.reshape(outputs, [-1, state_size])
    logits = tf.matmul(rnn_outputs, W) + b
    """
    predictions 아래 배열이 30개
    [[0.01222604  0.04997737  0.04972159  0.02963496  0.06532403  0.01507907
      0.02882861  0.06908683  0.0435047   0.03508175  0.04119922  0.04200007
      0.02193074  0.01616841  0.05267628  0.01069011  0.01190898  0.01439774
      0.02210629  0.02432451  0.02950356  0.01257133  0.01803722  0.02188681
      0.03182756  0.02962701  0.01342335  0.09229746  0.06643748  0.0285209]
      ...
     [0.01226655  0.03599569  0.04708919  0.0236398   0.06209319  0.01507676
      0.02982835  0.06306332  0.03819988  0.03158398  0.037941    0.04589246
      0.0254187   0.02194341  0.06868148  0.01146401  0.0117517   0.01561424
      0.02161234  0.02854498  0.02874204  0.01285549  0.02102005  0.02226738
      0.02811229  0.02931884  0.01322909  0.09467893  0.06808507  0.03398973]]
     """
    predictions = tf.nn.softmax(logits)
    # targets = [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 29  1  2  3  4  5 6  7  8  9 10]
    targets = tf.reshape(y_, [-1])

    #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=predictions)

    seq_weights = tf.ones([batchSize * seqLen])
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])

    loss = tf.reduce_mean(losses) / batchSize
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate, name="optimizer").minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())

    for i in range(epochNum):
        # get batches
        batchX, batchY = trainSet.__next__()
        # run train op
        feed_dict = {x_: batchX, y_: batchY}
        _, train_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

        if i % 1000 == 0:
            print('>> [{}] Average train loss : {}'.format(i, train_loss))
            # save model to disk
            saver.save(sess, ckpt_path + model_name + '.ckpt', global_step=i)

            # stop condidtion
            if train_loss < 0.01:
                print('\n>> Loss {}; Stopping training here at iteration #{}!!'.format(train_loss, i))
                break

            train_loss = 0


def predict(initChar, numChars, batchSize, seqLen, numClasses, stateSize):
    # reset graph
    tf.reset_default_graph()

    # placeholders
    x_ = tf.placeholder(tf.int64, [None, seqLen], name='x')
    y_ = tf.placeholder(tf.int64, [None, seqLen], name='y')

    embs = tf.get_variable('emb', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embs, x_)

    # parameters for softmax layer
    W = tf.Variable(tf.truncated_normal([state_size, numClasses], stddev=math.sqrt(2.0 / (seqLen))))
    b = tf.Variable(tf.random_uniform([numClasses], -1.0, 1.0))

    # rnn cell
    cell = rnn.GRUCell(stateSize)
    init_state = cell.zero_state(batchSize, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)
    rnn_outputs = tf.reshape(outputs, [-1, state_size])
    logits = tf.matmul(rnn_outputs, W) + b
    """
    predictions 아래 배열이 30개
    [[0.01222604  0.04997737  0.04972159  0.02963496  0.06532403  0.01507907
      0.02882861  0.06908683  0.0435047   0.03508175  0.04119922  0.04200007
      0.02193074  0.01616841  0.05267628  0.01069011  0.01190898  0.01439774
      0.02210629  0.02432451  0.02950356  0.01257133  0.01803722  0.02188681
      0.03182756  0.02962701  0.01342335  0.09229746  0.06643748  0.0285209]
      ...
     [0.01226655  0.03599569  0.04708919  0.0236398   0.06209319  0.01507676
      0.02982835  0.06306332  0.03819988  0.03158398  0.037941    0.04589246
      0.0254187   0.02194341  0.06868148  0.01146401  0.0117517   0.01561424
      0.02161234  0.02854498  0.02874204  0.01285549  0.02102005  0.02226738
      0.02811229  0.02931884  0.01322909  0.09467893  0.06808507  0.03398973]]
     """
    predictions = tf.nn.softmax(logits)
    # targets = [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 29  1  2  3  4  5 6  7  8  9 10]
    targets = tf.reshape(y_, [-1])

    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=predictions)

    seq_weights = tf.ones([batchSize * seqLen])
    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])

    loss = tf.reduce_mean(losses) / batchSize
    #optimizer = tf.train.AdamOptimizer(learning_rate=learningRate, name="optimizer").minimize(loss)


    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())

    # restore session
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    current_char = initChar
    chars = [current_char]

    for i in range(numChars):
        feed_dict = {x_: [[current_char]]}
        predv = sess.run([predictions], feed_dict=feed_dict)

        # sample from vocabulary
        #current_char = np.random.choice(predv.shape[-1], 1, p=np.squeeze(predv))[0]
        current_char = np.argmax(predv)
        chars.append(current_char)

    msg = ''.join([ idx2ch[chidx] for chidx in chars ])

    return msg



ckpt_path = '../resources/model/'
model_name = 'tf_lm'

num_layers = 2
state_size = 128
epochs = 100000000
learning_rate = 0.01


chars = ['!','#','$','@','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#idx2ch = list(sorted(set('\n'.join(chars))))
idx2ch = list(sorted(set(chars)))
ch2idx = {k: v for v, k in enumerate(idx2ch)}

X, Y = to_array(chars, ch2idx, seqLen=10)
trainset = rand_batch_gen(X, Y, batchSize=3)

# build the model
num_classes = len(idx2ch)

# train
train(trainset, learningRate=learning_rate, epochNum=epochs, batchSize=3, seqLen=X.shape[-1], numClasses=num_classes, stateSize=state_size)


# generate
init_char_idx = ch2idx['h']
num_chars = 10

msg = predict(initChar=init_char_idx, numChars=num_chars, batchSize=1, seqLen=1, numClasses=num_classes, stateSize=state_size)
print(msg)