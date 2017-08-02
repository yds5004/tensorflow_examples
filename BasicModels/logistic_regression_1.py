import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 10

# Training Data
batch_xs = np.asarray([
                            [3.3,4.4,5.5,6.71,6.93,5.4],
                            [4.168,9.779,5.23,6.182,7.59,2.3],
                            [2.167,7.042,10.791,5.313,7.997,6.76],
                            [1.23,2.54,6.43,7.65,2.3,7.6],
                            [5.654,9.27,3.1,5.2,8.2,3.2]
                        ])
batch_ys = np.asarray([
                            [1,0,0,0,0],
                            [0,1,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,1,0],
                            [0,0,0,0,1]
                        ])
n_samples = batch_xs.shape[0]
print (n_samples)

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 6]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 5]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([6, 5]))
b = tf.Variable(tf.zeros([5]))

# Construct model
prediction = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 5

        for (x_data, y_data) in zip(batch_xs, batch_ys):
            _, c = sess.run([optimizer, cost], feed_dict={x: [x_data], y: [y_data]})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: [x_data], y: [y_data]}))

    # prediction
    result1, result2 = sess.run([prediction, tf.argmax(prediction, 1)], feed_dict={x: batch_xs, y: batch_ys})
    print(result1)
    print(result2)
    sess.close()