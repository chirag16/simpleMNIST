import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Input data params
img_height = 28
img_width = 28
img_channels = 1
num_classes = 10

# Training params
learning_rate = 0.003
batch_size = 100
num_epochs = 1000

# Get the data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

X = tf.placeholder(tf.float32, [None, img_height * img_width])
W = tf.Variable(tf.zeros([img_height * img_width, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

# Model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# Placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# Loss function - Cross Entropy
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Create the session
sess = tf.Session()

# Initialize the variables
sess.run(tf.initialize_all_variables())

# Visualization
x_axis = []
training_accuracy_record = []
training_loss_record = []
testing_accuracy_record = []
testing_loss_record = []

# Run our Model!!
for i in range(num_epochs):
	# Load the input data
	batch_X, batch_Y = mnist_data.train.next_batch(batch_size)
	train_data = {X: batch_X, Y_: batch_Y}

	# Train
	sess.run(train_step, feed_dict = train_data)

	# Accuracy on training data
	acc, loss = sess.run([accuracy, cross_entropy], feed_dict = train_data)

	# Trying it out on test data
	test_data = {X: mnist_data.test.images, Y_: mnist_data.test.labels}

	# Accuracy on test data
	acc_test, loss_test = sess.run([accuracy, cross_entropy], feed_dict = test_data)
	
	x_axis.append(i)
	training_accuracy_record.append(acc)
	training_loss_record.append(loss)
	testing_accuracy_record.append(acc_test)
	testing_loss_record.append(loss_test)

	# Print stuff
	print (acc, loss, acc_test, loss_test)

plt.plot(x_axis, training_accuracy_record, 'r-', x_axis, testing_accuracy_record, 'b-')
plt.show()
plt.plot(x_axis, training_loss_record, 'r-', x_axis, testing_loss_record, 'b-')
plt.show()	