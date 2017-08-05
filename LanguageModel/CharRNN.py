import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import sys


class CharRNN(object):

    def __init__(self, seqlen, num_classes,
                 num_layers, state_size, epochs,
                 learning_rate, batch_size, ckpt_path,
                 model_name='char_rnn'):

        # attach to object
        self.epochs = epochs
        self.state_size = state_size
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.seqlen = seqlen

        # construct graph
        def __graph__():
            # reset graph
            tf.reset_default_graph()

            # placeholders
            x_ = tf.placeholder(tf.int64, [None, seqlen], name='x')
            y_ = tf.placeholder(tf.int64, [None, seqlen], name='y')

            # embeddings
            embs = tf.get_variable('emb', [num_classes, state_size])
            rnn_inputs = tf.nn.embedding_lookup(embs, x_)

            # rnn cell
            #cell = rnn.LSTMCell(state_size, state_is_tuple=True)
            #cell = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
            cell = rnn.MultiRNNCell([rnn.LSTMCell(state_size, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
            init_state = cell.zero_state(batch_size, tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, initial_state=init_state)

            # rnn_outputs.shape => [ batch_size, seqlen, state_size ]
            #  change to [ batch_size, seqlen x state_size ]
            rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])

            # parameters for softmax layer
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

            # output for each time step
            logits = tf.matmul(rnn_outputs, W) + b
            predictions = tf.nn.softmax(logits)

            # requires unnormalized prob
            y_flat = tf.reshape(y_, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_flat, logits=logits)
            loss = tf.reduce_mean(losses)

            # train op
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            # attach symbols to object, to expose to user of class
            self.x = x_
            self.y = y_
            self.init_state = init_state
            self.train_op = train_op
            self.loss = loss
            self.predictions = predictions
            self.final_state = final_state

        # run build graph
        __graph__()

    def train(self, train_set, epochs=None):

        epochs = self.epochs if not epochs else epochs

        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            # restore session
            if ckpt and ckpt.model_checkpoint_path:
                sys.stdout.write('\nrestoring saved model : {}\n\n'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            train_loss = 0
            for i in range(epochs):
                try:
                    # get batches
                    batchX, batchY = train_set.__next__()
                    # run train op
                    feed_dict = {self.x: batchX, self.y: batchY}
                    _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    sys.stdout.write('\r[{}/1000]'.format(1 + (i % 1000)))

                    # append to losses
                    train_loss += train_loss_
                    if i and i % 1000 == 0:
                        print('\n>> Average train loss : {}\n'.format(train_loss / 1000))
                        # save model to disk
                        saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)

                        # stop condidtion
                        #if (train_loss / 1000) < 0.5:
                        if (train_loss / 1000) < 1.5:
                            print(
                                '\n>> Loss {}; Stopping training here at iteration #{}!!'.format(train_loss / 1000, i))
                            break

                        train_loss = 0


                except KeyboardInterrupt:
                    print('\n>> Interrupted by user at iteration #' + str(i))
                    break

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    def predict(self, sess, X):
        predv = sess.run(self.predictions, feed_dict={self.x: X})
        # return the index of item with highest probability
        return np.argmax(predv, axis=1)

    def generate_characters(self, num_chars, init_char_idx):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            # init op
            sess.run(tf.global_variables_initializer())

            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            state = None
            current_char = init_char_idx
            chars = [current_char]

            for i in range(num_chars):
                if state:
                    feed_dict = {self.x: [[current_char]], self.init_state: state}
                else:
                    feed_dict = {self.x: [[current_char]]}

                predv, state = sess.run([self.predictions, self.final_state], feed_dict=feed_dict)

                # sample from vocabulary
                current_char = np.random.choice(predv.shape[-1], 1, p=np.squeeze(predv))[0]
                chars.append(current_char)

        return chars