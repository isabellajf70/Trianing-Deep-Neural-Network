import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 


if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()    

# Load MNIST dataset
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


learningRate = 0.001
trainingIters = 100000
batchSize = 256
displayStep = 20

nInput = 28#we want the input to take the 28 pixels
nSteps = 28#every 28
nHidden = 256#number of neurons for the RNN
nClasses = 10#this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

	lstmCell = rnn_cell.GRUCell(nHidden)#find which lstm to use in the documentation

	outputs, states = rnn.static_rnn(lstmCell, x, dtype = tf.float32)#for the rnn where to get the output and hidden state 

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correctPred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
result_dir = './rnn_other/gru_rnn_hidden=256/'
summary_op = tf.summary.merge_all()
test_acc = tf.summary.scalar('Test_Accuracy', accuracy)
train_acc = tf.summary.scalar('Train_Accuracy', accuracy)
trainloss = tf.summary.scalar('Train_Loss', cost)
with tf.Session() as sess:
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()
	
	summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

	sess.run(init)
	step = 1

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict = {x: batchX, y:batchY})
			loss = sess.run(cost, feed_dict = {x:batchX, y:batchY})
			test_accuracy = sess.run(test_acc, feed_dict={x: np.reshape(mnist.test.images, [-1, 28, 28]), y: mnist.test.labels})
			train_accuracy = sess.run(train_acc, feed_dict={x: batchX, y: batchY})
			train_loss = sess.run(trainloss, feed_dict={x: batchX, y: batchY})
			summary_writer.add_summary(test_accuracy, step)
			summary_writer.add_summary(train_accuracy, step)
			summary_writer.add_summary(train_loss, step)

			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
		step +=1
	print('Optimization finished')

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))