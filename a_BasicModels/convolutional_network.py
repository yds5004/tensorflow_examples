import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("rsc/MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.get_variable("wc1", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer()),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable("wc2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.get_variable("wd1", shape=[7*7*64, 1024], initializer=tf.contrib.layers.xavier_initializer()),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.get_variable("out", shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def connectFully(x, W, b):
    fc1 = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, W), b)
    fc1 = tf.nn.relu(fc1)

    return tf.nn.dropout(fc1, dropout)



# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = connectFully(pool2, weights['wd1'], biases['bd1'])

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop)
    session.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
    if step % display_step == 0:
        # Calculate batch loss and accuracy
        loss, acc = session.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " +  "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))
    step += 1
print ("Optimization Finished!")

# Calculate accuracy for 256 mnist test images
print ("Testing Accuracy:")
session.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
result1, result2 = session.run([pred, tf.arg_max(pred, 1)], feed_dict={x: mnist.test.images[:256], keep_prob: 1.})
print (result1)
print (result2)