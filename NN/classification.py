import tensorflow as tf

input_data = [[ 1, 5, 3, 7, 8, 10, 12 ], [5, 8, 10, 3, 9, 7, 1]]
label_data = [[ 0, 0, 0, 1, 0 ], [ 1, 0, 0, 0, 0 ]]

CLASS_SIZE = 5
LEARNING_RATE = 0.01
training_iters = 1000
display_iters = 100

INPUT_SIZE = 7
HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 128


x = tf.placeholder( tf.float32, shape=[None, INPUT_SIZE], name='x')
y_ = tf.placeholder( tf.float32, shape=[None, CLASS_SIZE], name='y')

weights = {
    'out1': tf.get_variable("out1", shape=[INPUT_SIZE, HIDDEN1_SIZE], initializer=tf.contrib.layers.xavier_initializer()),
    'out2': tf.get_variable("out2", shape=[HIDDEN1_SIZE, HIDDEN2_SIZE], initializer=tf.contrib.layers.xavier_initializer()),
    'out3': tf.get_variable("out3", shape=[HIDDEN2_SIZE, CLASS_SIZE], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'out1': tf.Variable(tf.constant(0.1, shape=[HIDDEN1_SIZE])),
    'out2': tf.Variable(tf.constant(0.1, shape=[HIDDEN2_SIZE])),
    'out3': tf.Variable(tf.constant(0.1, shape=[CLASS_SIZE]))
}

hidden1 = tf.nn.relu( tf.matmul( x, weights['out1']) + biases['out1'] )
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights['out2']) + biases['out2'] )
y = tf.nn.softmax(tf.matmul(hidden2, weights['out3']) + biases['out3'] )

cost1 = -y_ * tf.log(y) - (1 - y_) * tf.log(1 - y)
cost2 = tf.reduce_sum(cost1, reduction_indices=[1])
cost3 = tf.reduce_mean(cost2)

train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost3)
#train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost3)

compare_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(compare_prediction, tf.float32))

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

feed = { x: input_data, y_: label_data }
for i in range (training_iters):
    _, cost, acc = session.run ( [train, cost3, accuracy], feed_dict=feed )
    if i % display_iters == 0:
        print ('step: ', i, ' / loss: ', cost, ' / acc: ', acc)

result1, result2 = session.run ( [y, tf.arg_max(y, 1)], feed_dict=feed )
print (result1)
print (result2)