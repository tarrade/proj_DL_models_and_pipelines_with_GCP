"""
Basic ideas frm https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py
- update using tf.data
- update session
"""
import tensorflow as tf
import numpy as np


NUM_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.1


###############################################################################
# # Decide if you want eager execution:
# EAGER = True
# if EAGER:
#     tf.enable_eage_execution() #  ['config=None', 'device_policy=None', 'execution_mode=None']
try:
    raise Exception
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (X_test, y_test) = mnist.load_data()
except Exception:
  print("download manually to ./data/ from {}".format(
      "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
  ))
  with np.load("./data/mnist.npz") as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# classic numpy approach using reshape and
# reshape and save image dimensions
dim_img = x_train.shape[1:]
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

# Convert Numpy Array to Tensor manually to avoid accidential reassignment:
# x_train = tf.convert_to_tensor(x_train, name = None)
# x_test  = tf.convert_to_tensor(x_test, name = None)

print("passed")


def oneHotEncode(array):
    n = len(array)
    dense_array = np.zeros((n, len(set(array))))
    dense_array[np.arange(n), array] = 1
    return dense_array


assert set(y_train) == set(
    y_test), "Classes in train and test set are different. which is correct?"
classes = set(y_train)  # 0-9 digits
y_train = oneHotEncode(y_train)
y_test = oneHotEncode(y_test)
###############################################################################
# parser function for input data

# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.


def parser(record):
    """
    Define a function to pass to map fct of tf.data.Dataset
    example: https://www.tensorflow.org/guide/datasets#using_high-level_apis
    """
    # keys_to_features = {
    #     "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
    #     "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
    #     "label": tf.FixedLenFeature((), tf.int64,
    #                                 default_value=tf.zeros([], dtype=tf.int64)),
    # }

    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.float, default_value=""),
        "label": tf.FixedLenFeature((), tf.int32,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label
###############################################################################
# Build Model


# data specific parameters:
num_classes = y_train.shape[-1]  # MNIST total classes (0-9 digits)
num_input = x_train.shape[-1]  # MNIST data input (img shape: 28*28)


# Number of units in hidden layer (Hyperparameter)
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons


# tf Graph input
X = tf.placeholder("float", shape=[None, num_input])
Y = tf.placeholder("int32", shape=[None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model


def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# # Using tf.layers:
# tf.layers.dense(X, activation= tf... relu, units= n_hidden_1)


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()  # ToDo: depreciated


##############################################################################
#
# tf.data , Preprocessing

# Load from Numpy, strategy: https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays
trainslices = tf.data.Dataset.from_tensor_slices((X, Y))
trainslices = trainslices.shuffle(buffer_size=3000,
                                  seed=123456,
                                  reshuffle_each_iteration=True)
# trainslices = trainslices.map(parser) # to add to tf.data pipeline
trainslices = trainslices.repeat(count=1)
trainslices = trainslices.batch(batch_size=BATCH_SIZE,
                                drop_remainder=True)  # if False -> breaks assert in training loop
iterator = trainslices.make_initializable_iterator()
next_element = iterator.get_next()

# tf.data.experimental.make_batched_features_dataset(BATCH_SIZE, traindata, num_epochs=NUM_EPOCHS)

# #unified call possible (unclear how to to do with numpy arrays)
# iterator = traindata.make_initializable_iterator(
#                                  batch_size=BATCH_SIZE,
#                                  features=traindata,
#                                  num_epochs=NUM_EPOCHS)

testslices = tf.data.Dataset.from_tensor_slices((X, Y))

# Sequencing batch to mini-batches
# https://github.com/tensorflow/tensorflow/blob/9230423668770036179a72414482d45ddde40a3b/tensorflow/contrib/training/python/training/sequence_queueing_state_saver.py#L353
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    # sess.run(init)
    sess.run(init)
    print('Initalized graph')
    for _ in range(NUM_EPOCHS):
        # i = 0
        # print("#### Epoch {}".format(_))
        sess.run(iterator.initializer, feed_dict={X: x_train,
                                                  Y: y_train})
        while True:
            try:
                # i += 1
                # if i % 100 == 0:
                #     print("# batch {}".format(i))
                batch_x, batch_y = sess.run(next_element)
                assert batch_x.shape == (
                    BATCH_SIZE, num_input), "Something is wrong with Batch shape: {}".format(batch_x.shape)
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            except tf.errors.OutOfRangeError:
                # Calculate metrics for validation set / test set
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train,
                                                                     Y: y_train})
                print("Epoch " + str(_ + 1) + ", Training Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))
                break

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: x_test,
                                        Y: y_test}))
