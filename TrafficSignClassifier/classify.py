# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
import sklearn.utils

training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

n_train = len(train['features'])
n_validation = len(valid['features'])
n_test = len(test['features'])

image_shape = np.shape(train['features'])[1:]
n_classes = len(set(train["labels"]))

# It is required to normalize the data so that it has a mean of zero and equal variance.
X_train = (np.array(train['features']) + -128) / 128
X_valid = (np.array(valid['features']) + -128) / 128
X_test = (np.array(test['features']) + -128) / 128

y_train = train['labels']
y_valid = valid['labels']
y_test = test['labels']

def weight_variable(dimensions, mu, sigma):
    return tf.Variable(tf.truncated_normal(shape=dimensions, mean=mu, stddev=sigma))

def bias_variable(output_depth):
    return tf.Variable(tf.zeros(output_depth))

def convolutional_layer(x, input_depth, output_depth, mu, sigma):
    weights = weight_variable((5, 5, input_depth, output_depth), mu, sigma)
    biases = bias_variable(output_depth)
    layer = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID') + biases

    return tf.nn.relu(layer)

def pooling_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def connected_layer(x, input_depth, output_depth, mu, sigma, keep_prob, is_activated=True):
    weights = weight_variable((input_depth, output_depth), mu, sigma)
    biases = bias_variable(output_depth)
    layer = tf.matmul(x, weights) + biases

    return tf.nn.dropout(tf.nn.relu(layer), keep_prob) if is_activated else layer

keep_prob = tf.placeholder(tf.float32)

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Input Layer = 32x32x3

    # Layer 1 = 28x28x6
    layer1 = convolutional_layer(x, 3, 6, mu, sigma)
    # Pooling = 14x14x6
    layer1 = pooling_layer(layer1)

    # Layer 2 = 10x10x16
    layer2 = convolutional_layer(layer1, 6, 16, mu, sigma)
    # Pooling = 5x5x16
    layer2 = pooling_layer(layer2)
    # Flatten = 400
    layer2 = tf.confib.layers.flatten(layer2)

    # Layer 3 = 120
    layer3 = connected_layer(layer2, 400, 120, mu, sigma, keep_prob)

    # Layer 4 = 84
    layer4 = connected_layer(layer3, 120, 84, mu, sigma, keep_prob)

    # Layer 5 = Number of classes
    layer5 = connected_layer(layer4, 84, n_classes, mu, sigma, keep_prob, is_activated=False)

    return layer5

# Train the model.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entrop_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    session = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = session.run(accuracy_operation, feed_dict={
            x: batch_x,
            y: batch_y,
            keep_prob: 1.0,
        })
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

EPOCHS = 15
BATCH_SIZE = 128
KEEP_PROB = 0.5

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")

    for epoch_index in range(EPOCHS):
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: KEEP_PROB,
            })

        print("EPOCH {}".format(epoch_index + 1))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    training_accuracy = evaluate(X_train, y_train)
    print("Training Accuracy = {:.3f}".format(training_accuracy))
    saver.save(session, './traffic_sign')
    print("Model saved")

with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
