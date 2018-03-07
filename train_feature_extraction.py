import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time
from sklearn.utils import shuffle

nb_classes = 43
learning_rate = 0.01
batch = 128
epoch = 10
# TODO: Load traffic signs data.
training_file = 'train.p'
with open(training_file, mode='rb') as f:
	train_data = pickle.load(f)
X_data, y_data = train_data['features'], train_data['labels']
# TODO: Split data into training and validation sets.
X_train,X_valid,y_train,y_valid = train_test_split(X_data,y_data,test_size=0.33,random_state=0)
# TODO: Define placeholders and resize operation.
X = tf.placeholder(tf.float32, (None,32,32,3))
y = tf.placeholder(tf.int64,(None))
X_resized = tf.image.resize_images(X,(227,227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(X_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
final_layer_w = tf.Variable(tf.truncated_normal(shape,stddev = 0.01))
final_layer_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7,final_layer_w) + final_layer_b
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation, var_list = [final_layer_w, final_layer_b])

correct_predict = tf.equal(tf.argmax(logits,1), y)
accuracy_opertation = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

def evaluate(X_input,y_input):
	num_examples = len(X_input)
	total_accuracy = 0
	for offset in range(0, num_examples, batch):
		batch_X, batch_y = X_input[offset:offset+batch], y_input[offset:offset+batch]
		accuracy = sess.run(accuracy_opertation, feed_dict={X: batch_X, y: batch_y})
		total_accuracy += (accuracy * len(batch_X))
	return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(epoch):
		t0 = time.time()
		X_train, y_train = shuffle(X_train,y_train)
		for offset in range(0, X_train.shape[0], batch):
			end = offset + batch
			sess.run(training_operation, feed_dict = {X: X_train[offset:end], y: y_train[offset:end]})

		valid_acc = evaluate(X_valid,y_valid)
		print("Epoch", i+1)
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation accuracy:", valid_acc)
		print("")