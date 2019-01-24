"""
Basic ideas frm https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py
- update using tf.data
- update session
"""
import tensorflow as tf
import numpy as np

NUM_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
###############################################################################
#Load Data
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
# x_train = tf.cast(x_train, dtype="float")
# x_test  = tf.cast(x_test, dtype="float")
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
dim_input = x_train.shape[-1]  # MNIST data input (img shape: 28*28)

# Number of units in hidden layer (Hyperparameter)
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons

# tf Graph input
X = tf.placeholder("float", shape=[None, dim_input])
Y = tf.placeholder("int32", shape=[None, num_classes])



# # Create model a statful model
class FNN(object):
  def __init__(self):  
    self.w_1 = tf.Variable(tf.Variable(tf.random_normal([dim_input, n_hidden_1])), name='W1')
    self.b_1 = tf.Variable(tf.random_normal([n_hidden_1]), name='b1')
    self.w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2")
    self.b_2 = tf.Variable(tf.random_normal([n_hidden_2]), name='b2')
    self.w_out = tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name="W_out")
    self.b_out = tf.Variable(tf.random_normal([num_classes]), name='b_out')
    #self.weights = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_out, self.b_out]
  
  def __call__(self, inputs, training=False):
    hidden_1 = tf.nn.relu(tf.matmul(inputs, self.w_1) + self.b_1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w_2) + self.b_2)
    logits =   tf.matmul(hidden_2, self.w_out) + self.b_out
    return logits

# class FNN(tf.keras.Model):
#   def __init__(self):
#     super(FNN, self).__init__()
#     self.w_1 = tf.Variable(tf.Variable(tf.random_normal([dim_input, n_hidden_1])), name='W1')
#     self.b_1 = tf.Variable(tf.random_normal([n_hidden_1]), name='b1')
#     self.w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2")
#     self.b_2 = tf.Variable(tf.random_normal([n_hidden_2]), name='b2')
#     self.w_out = tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name="W_out")
#     self.b_out = tf.Variable(tf.random_normal([num_classes]), name='b_out')
#     #self.weights = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_out, self.b_out]
  
#   def call(self, inputs, training=False):
#     hidden_1 = tf.nn.relu(tf.matmul(inputs, self.w_1) + self.b_1)
#     hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w_2) + self.b_2)
#     logits =   tf.matmul(hidden_2, self.w_out) + self.b_out
#     return logits

# Construct model

logits = FNN()(X)


# Construct model

logits = FNN()(X)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#     logits=logits, labels=Y))

loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
# train_op = optimizer.minimize(loss_op)

# Evaluate model
tp = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(tp, dtype="float"))

# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()  # ToDo: depreciated
##############################################################################
# tf.data , Preprocessing
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


N_train = x_train.shape[0]
n_batches = N_train / BATCH_SIZE
step = int(n_batches/ 60)

init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    # sess.run(init)
    sess.run(init)
    print('Initalized graph')
    for i in range(1, NUM_EPOCHS+1):
        print("Epoch {}: ".format(i), end='')
        sess.run(iterator.initializer, feed_dict={X: x_train,
                                                  Y: y_train})
        batch = 0
        while True:
            try:
                images, labels = sess.run(next_element)
                assert images.shape == (
                    BATCH_SIZE, dim_input), "Something is wrong with Batch shape: {}".format(images.shape)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={X: images, Y: labels})
                if batch % step == 0:
                    print('#', end='')
                batch += 1
            except tf.errors.OutOfRangeError:
                # Calculate metrics for validation set / test set
                _loss, _acc = sess.run([loss, accuracy], feed_dict={X: x_train,
                                                                       Y: y_train})
                print(", Training Loss= {:.2f}".format(_loss) +
                      ", Training Accuracy= {:.3f}".format(_acc))
                break
        print("Validation Accuracy:",
          sess.run(accuracy, feed_dict={X: x_test,
                                        Y: y_test}))


    # print("Optimization Finished!")

    # # Calculate accuracy for MNIST test images
    # print("Validation Accuracy:",
    #       sess.run(accuracy, feed_dict={X: x_test,
    #                                     Y: y_test}))
