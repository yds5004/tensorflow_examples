import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

CLASS_1 = 1  # next is space
CLASS_0 = 0  # next is not space
class_size = 2

# "hi, hello"
# {' ': 0, ',': 1, 'e': 2, 'h': 3, 'i': 4, 'l': 5, 'o': 6}
# ,:[0,1,0,0,0,0,0] e:[0,0,1,0,0,0,0], h:[0,0,0,1,0,0,0], i:[0,0,0,0,1,0,0], l:[0,0,0,0,0,1,0], o:[0,0,0,0,0,0,1]
x_data = [
            [ [0, 0, 0, 1, 0, 0, 0],  # h
              [0, 0, 0, 0, 1, 0, 0],  # i
              [0, 1, 0, 0, 0, 0, 0],  # ,
              [0, 0, 0, 1, 0, 0, 0],  # h
              [0, 0, 1, 0, 0, 0, 0],  # e
              [0, 0, 0, 0, 0, 1, 0],  # l
              [0, 0, 0, 0, 0, 1, 0],  # l
              [0, 0, 0, 0, 0, 0, 1] ] # o
        ]
y_data = [[0, 0, 1, 0, 0, 0, 0, 1]]

batch_x = np.array(x_data, dtype='f')
batch_y = np.array(y_data, dtype='int32')


# Parameters
learning_rate = 0.001
training_iters = 500
display_step = 100

# Network Parameters
batch_size = 1
sequence_size = 8 # timesteps
vector_size = 7 # vocaburary size
hidden_size1 = 256  # hidden layer num of features
hidden_size2 = 64  # hidden layer num of features
hidden_size3 = 16  # hidden layer num of features
drop_out_rate = 0.7

# tf Graph input
x = tf.placeholder("float", [None, sequence_size, vector_size])
y = tf.placeholder("int32", [None, sequence_size])

weights = {
    'out1': tf.get_variable("out1", shape=[2 * hidden_size1, hidden_size2], initializer=tf.contrib.layers.xavier_initializer()),
    'out2': tf.get_variable("out2", shape=[hidden_size2, hidden_size3], initializer=tf.contrib.layers.xavier_initializer()),
    'out3': tf.get_variable("out3", shape=[hidden_size3, class_size], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'out1': tf.Variable(tf.constant(0.1, shape=[hidden_size2]), name="bias_out1"),
    'out2': tf.Variable(tf.constant(0.1, shape=[hidden_size3]), name="bias_out2"),
    'out3': tf.Variable(tf.constant(0.1, shape=[class_size]), name="bias_out3")
}

def BidirectionalRNN(X, weights, biases):
    X = tf.unstack(X, sequence_size, axis=1)

    fw_cell = rnn.GRUCell(hidden_size1)
    bw_cell = rnn.GRUCell(hidden_size1)

    outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, X, dtype=tf.float32)

    final_outputs = []
    for output in outputs:
        hidden1   = tf.nn.relu( tf.matmul( output, weights['out1']) + biases['out1'] )
        hidden1_1 = tf.nn.dropout(hidden1, drop_out_rate)
        hidden2   = tf.nn.relu( tf.matmul( hidden1_1, weights['out2']) + biases['out2'] )
        hidden2_1 = tf.nn.dropout(hidden2, drop_out_rate)
        out       = tf.nn.relu( tf.matmul(hidden2_1, weights['out3']) + biases['out3'] )
        final_outputs.append(out)
    return final_outputs

y_ = BidirectionalRNN(x, weights, biases)
logits = tf.reshape(tf.concat(y_, 1), [-1, class_size], name="logits")
targets = tf.reshape(y, [1, -1], name="targets")

seq_weights = tf.ones([batch_size * sequence_size])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)



NUM_THREADS = 1
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

step = 0
while step < training_iters :
    feed={x: batch_x, y: batch_y}
    sess.run(optimizer, feed_dict=feed)

    if step % display_step == 0 :
        p_x, p_y, p_y_, p_logits, p_cost = sess.run([x, y, y_, logits, cost], feed_dict=feed)
        print ('step : %s' % step + ',' + 'cost : %s' % p_cost)

    step += 1


# inference
test_sentences = ['hi,hello']

i = 0
while i < len(test_sentences) :
    sentence = test_sentences[i]
    length = len(sentence)
    diff = sequence_size - length
    if diff > 0 : # add padding
        test_sentences[i] += ' '*diff
    i += 1

batch_size = len(test_sentences)

feed={x: batch_x, y: batch_y}
y_, logits, result = sess.run([y_, logits, tf.arg_max(logits, 1)], feed_dict=feed)

i = 0
while i < len(test_sentences) :
    sentence = test_sentences[i]
    bidx = i * sequence_size
    eidx = bidx + sequence_size
    rst = result[bidx:eidx]

    out = []
    j = 0
    while j < sequence_size :
        tag = rst[j]
        if tag == CLASS_1 :
            out.append(sentence[j])
            out.append(' ')
        else :
            out.append(sentence[j])
        j += 1
    n_sentence = ''.join(out).strip()
    print ('out = ' + n_sentence)
    i += 1
sess.close()