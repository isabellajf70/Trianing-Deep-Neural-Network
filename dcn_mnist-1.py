import os
import time

import tensorflow as tf

if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

# Load MNIST dataset
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()


def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE

    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max

def main():
    # Specify training parameters
    result_dir = './results/' # directory where the results from the training are saved
    max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

    # placeholders for input data and input labeles
    x = tf.compat.v1.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y_')

    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)
    # Build the summary operation based on the TF collection of Summaries.

    tf.summary.scalar('conv1_weight_mean', tf.reduce_mean(W_conv1))
    tf.summary.scalar('conv1_weight_max', tf.reduce_max(W_conv1))
    tf.summary.scalar('conv1_weight_min', tf.reduce_min(W_conv1))
    tf.summary.scalar('conv1_weight_std', (tf.reduce_sum(W_conv1 - tf.reduce_mean(W_conv1))) ** 2)
    tf.summary.histogram('conv1_weight_hist', W_conv1)

    tf.summary.scalar('conv1_bias_mean', tf.reduce_mean(b_conv1))
    tf.summary.scalar('conv1_bias_max', tf.reduce_max(b_conv1))
    tf.summary.scalar('conv1_bias_min', tf.reduce_min(b_conv1))
    tf.summary.scalar('conv1_bias_std', (tf.reduce_sum(b_conv1 - tf.reduce_mean(b_conv1))) ** 2)
    tf.summary.histogram('conv1_bias_hist', b_conv1)

    tf.summary.scalar('conv1_input_mean', tf.reduce_mean(x_image))
    tf.summary.scalar('conv1_input_max', tf.reduce_max(x_image))
    tf.summary.scalar('conv1_input_min', tf.reduce_min(x_image))
    tf.summary.scalar('conv1_input_std', (tf.reduce_sum(x_image - tf.reduce_mean(x_image))) ** 2)
    tf.summary.histogram('conv1_input_hist', x_image)

    tf.summary.scalar('conv1_activation_mean', tf.reduce_mean(h_conv1))
    tf.summary.scalar('conv1_activation_max', tf.reduce_max(h_conv1))
    tf.summary.scalar('conv1_activation_min', tf.reduce_min(h_conv1))
    tf.summary.scalar('conv1_activation_std', (tf.reduce_sum(h_conv1 - tf.reduce_mean(h_conv1))) ** 2)
    tf.summary.histogram('conv1_activation_hist', h_conv1)

    tf.summary.scalar('conv2_weight_mean', tf.reduce_mean(W_conv2))
    tf.summary.scalar('conv2_weight_max', tf.reduce_max(W_conv2))
    tf.summary.scalar('conv2_weight_min', tf.reduce_min(W_conv2))
    tf.summary.scalar('conv2_weight_std', (tf.reduce_sum(W_conv2 - tf.reduce_mean(W_conv2))) ** 2)
    tf.summary.histogram('conv2_weight_hist', W_conv2)

    tf.summary.scalar('conv2_bias_mean', tf.reduce_mean(b_conv2))
    tf.summary.scalar('conv2_bias_max', tf.reduce_max(b_conv2))
    tf.summary.scalar('conv2_bias_min', tf.reduce_min(b_conv2))
    tf.summary.scalar('conv2_bias_std', (tf.reduce_sum(b_conv2 - tf.reduce_mean(b_conv2))) ** 2)
    tf.summary.histogram('conv2_bias_hist', b_conv2)

    tf.summary.scalar('conv2_input_mean', tf.reduce_mean(h_pool1))
    tf.summary.scalar('conv2_input_max', tf.reduce_max(h_pool1))
    tf.summary.scalar('conv2_input_min', tf.reduce_min(h_pool1))
    tf.summary.scalar('conv2_input_std', (tf.reduce_sum(h_pool1 - tf.reduce_mean(h_pool1))) ** 2)
    tf.summary.histogram('conv2_input_hist', h_pool1)

    tf.summary.scalar('conv2_activation_mean', tf.reduce_mean(h_conv2))
    tf.summary.scalar('conv2_activation_max', tf.reduce_max(h_conv2))
    tf.summary.scalar('conv2_activation_min', tf.reduce_min(h_conv2))
    tf.summary.scalar('conv2_activation_std', (tf.reduce_sum(h_conv2 - tf.reduce_mean(h_conv2))) ** 2)
    tf.summary.histogram('conv2_activation_hist', h_conv2)

    tf.summary.scalar('fc_bias_mean', tf.reduce_mean(b_fc1))
    tf.summary.scalar('fc_bias_max', tf.reduce_max(b_fc1))
    tf.summary.scalar('fc_bias_min', tf.reduce_min(b_fc1))
    tf.summary.scalar('fc_bias_std', (tf.reduce_sum(b_fc1 - tf.reduce_mean(b_fc1))) ** 2)
    tf.summary.histogram('fc_bias_hist', b_fc1)

    tf.summary.scalar('fc_weight_mean', tf.reduce_mean(W_fc1))
    tf.summary.scalar('fc_weight_max', tf.reduce_max(W_fc1))
    tf.summary.scalar('fc_weight_min', tf.reduce_min(W_fc1))
    tf.summary.scalar('fc_weight_std', (tf.reduce_sum(W_fc1 - tf.reduce_mean(W_fc1))) ** 2)
    tf.summary.histogram('fc_weight_hist', W_fc1)

    tf.summary.scalar('fc_input_mean', tf.reduce_mean(h_pool2_flat))
    tf.summary.scalar('fc_input_max', tf.reduce_max(h_pool2_flat))
    tf.summary.scalar('fc_input_min', tf.reduce_min(h_pool2_flat))
    tf.summary.scalar('fc_input_std', (tf.reduce_sum(h_pool2_flat - tf.reduce_mean(h_pool2_flat))) ** 2)
    tf.summary.histogram('fc_input_hist', h_pool2_flat)

    tf.summary.scalar('softmax_weight_mean', tf.reduce_mean(W_fc2))
    tf.summary.scalar('softmax_weight_max', tf.reduce_max(W_fc2))
    tf.summary.scalar('softmax_weight_min', tf.reduce_min(W_fc2))
    tf.summary.scalar('softmax_weight_std', (tf.reduce_sum(W_fc2 - tf.reduce_mean(W_fc2))) ** 2)
    tf.summary.histogram('softmax_weight_hist', W_fc2)

    tf.summary.scalar('softmax_bias_mean', tf.reduce_mean(b_fc2))
    tf.summary.scalar('softmax_bias_max', tf.reduce_max(b_fc2))
    tf.summary.scalar('softmax_bias_min', tf.reduce_min(b_fc2))
    tf.summary.scalar('softmax_bias_std', (tf.reduce_sum(b_fc2 - tf.reduce_mean(b_fc2))) ** 2)
    tf.summary.histogram('softmax_bias_hist', b_fc2)

    tf.summary.scalar('softmax_input_mean', tf.reduce_mean(h_fc1_drop))
    tf.summary.scalar('softmax_input_max', tf.reduce_max(h_fc1_drop))
    tf.summary.scalar('softmax_input_min', tf.reduce_min(h_fc1_drop))
    tf.summary.scalar('softmax_input_std', (tf.reduce_sum(h_fc1_drop - tf.reduce_mean(h_fc1_drop))) ** 2)
    tf.summary.histogram('softmax_input_hist', h_fc1_drop)

    summary_op = tf.summary.merge_all()
    validation_acc_summary = tf.summary.scalar('validation_accuracy', accuracy)
    test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(50)  # make the data batch, which is used in the training iteration.
        # the batch size is 50
        if i % 100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            test_accuracy = sess.run(test_acc_summary,
                                     feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            validation_accuracy = sess.run(validation_acc_summary,
                                           feed_dict={x: mnist.validation.images, y_: mnist.validation.labels,
                                                      keep_prob: 1.0})
            summary_writer.add_summary(test_accuracy, i)
            summary_writer.add_summary(validation_accuracy, i)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

    # print test error
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    stop_time = time.time()
    print('The training takes %f second to finish' % (stop_time - start_time))


if __name__ == "__main__":
    main()

