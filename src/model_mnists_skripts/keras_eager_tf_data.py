"""
Basic ideas frm https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py
- update using tf.data
- update session
"""
import tensorflow as tf
import numpy as np

# EAGER !!!
tf.enable_eager_execution()  #  ['config=None', 'device_policy=None', 'execution_mode=None']

NUM_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
###############################################################################
# Load Data
# data is in memory
try:
    with np.load("./data/mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']  
except Exception:
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (X_test, y_test) = mnist.load_data()
    except Exception:
        raise Exception("Not Connection to Server: Download manually to ./data/ from {}".format(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
         ))

# classic numpy approach using reshape and
# reshape and save image dimensions
dim_img = x_train.shape[1:]

# processing data is done using tf.data
"""
#######################################
# parser function for input data
def image_parser(array:np.array):
    "Return `tf.Tensor` with parsed images."
    array = tf.cast(array, dtype="float")
    array = tf.reshape(array, [array.shape[0].value, -1])
    return array

x_train = image_parser(x_train)
x_test  = image_parser(x_test)

print("Data Loaded in Memory")

def oneHotEncode(array):
    n = len(array)
    dense_array = np.zeros((n, len(set(array))))
    dense_array[np.arange(n), array] = 1
    return dense_array

assert set(y_train) == set(
    y_test), "Classes in train and test set are different. which is correct?"

y_train = oneHotEncode(y_train)
y_test = oneHotEncode(y_test)
"""
###############################################################################
# Build Model
classes = set(y_train)  # 0-9 digits
# data specific parameters:
num_classes = len(classes)  # MNIST total classes (0-9 digits)
dim_input = 1
for x in dim_img: dim_input *= x # MNIST data input (img shape: 28*28)
print("N target classes: {}, Flat-Image-Vector-Length: {}".format(num_classes, dim_input))

# Number of units in hidden layer (Hyperparameter)
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons

# Create model
class FNN(tf.keras.Model):
  def __init__(self):
    super(FNN, self).__init__()
    self.w_1 = tf.Variable(tf.Variable(tf.random_normal([dim_input, n_hidden_1])), name='W1')
    self.b_1 = tf.Variable(tf.random_normal([n_hidden_1]), name='b1')
    self.w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2")
    self.b_2 = tf.Variable(tf.random_normal([n_hidden_2]), name='b2')
    self.w_out = tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name="W_out")
    self.b_out = tf.Variable(tf.random_normal([num_classes]), name='b_out')
    #self.weights = [self.w_1, self.b_1, self.w_2, self.b_2, self.w_out, self.b_out]
  
  def call(self, inputs, training=False):
    hidden_1 = tf.nn.relu(tf.matmul(inputs, self.w_1) + self.b_1)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w_2) + self.b_2)
    logits =   tf.matmul(hidden_2, self.w_out) + self.b_out
    return logits

# Define loss and optimizer
def loss(model, inputs, targets):
    logits = model(tf.cast(inputs, dtype="float"))
    loss_value = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=targets)
    return loss_value

def acc(model, inputs, targets):
    logits = model(inputs)
    class_predicted = tf.argmax(input=logits, axis=1)
    tp = tf.equal(class_predicted, tf.argmax(targets, axis=1))
    acc = tf.reduce_mean(tf.cast(tp, dtype="float"))
    return acc

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.weights) # ToDo: Create List of variables in model definition

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
model = FNN()
##############################################################################
# tf.data, Preprocessing
# https://www.tensorflow.org/guide/performance/datasets

trainslices = tf.data.Dataset.from_tensor_slices((x_train, y_train))

trainslices = trainslices.shuffle(buffer_size=3000,
                                  seed=123456,
                                  reshuffle_each_iteration=True)

# #ToDo: Add reshape here. How to call whole dataset?
def _parse_fct(image, label):
    """Parse input data."""
    #parse image:    
    image = tf.reshape(image, [-1])
    image = tf.cast(image, dtype="float")
    # one-hot encode integers 0-9:
    dense_label = tf.one_hot(indices=label, depth=num_classes, dtype="int32")
    # dense_label = tf.reshape(dense_label, [1, -1])
    return image, dense_label 

trainslices = trainslices.map(map_func=_parse_fct, num_parallel_calls=3)


#trainslices = trainslices.repeat(count=1)

trainslices = trainslices.batch(batch_size=BATCH_SIZE,Berlin
                                drop_remainder=True)  # if False -> breaks assert in training loop

testslices = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# print("Initial loss: {:.3f}".format(
#     loss(model, trainslices.take(-1))))

# print("Initial acc: {:.3f}".format(
#     acc(model, x_train, y_train)))    

# loss_history = []

N_train = x_train.shape[0]
n_batches = int(N_train / BATCH_SIZE) #take floor of float, since remainder of dataset not fitting into dataset is dropped
step = int(n_batches/ 60)
for i in range(1, NUM_EPOCHS + 1):
    print("Epoch {}: ".format(i), end='')
    loss_epoch = 0
    acc_epoch = 0 
    for (batch, (images, labels)) in enumerate(trainslices.take(-1)):
        if batch % step == 0:
            print('#', end='')
        # optimizer.minimize has two steps in eager mode:Berlin  
        grads = grad(model, images, labels) #ToDo: add tf.cast to parser fct
        optimizer.apply_gradients(zip(grads, model.weights),
                                    global_step=tf.train.get_or_create_global_step())
        loss_epoch += loss(model, images, labels)
        acc_epoch += acc(model, images, labels)
    # _loss = loss(model, tf.cast(x_train, dtype="float"), y_train)
    # _acc  =  acc(model, tf.cast(x_train, dtype="float"), y_train)
    print(", Training Loss= {:.2f}"
        ", Training Accuracy= {:.3f}".format(loss_epoch/n_batches, acc_epoch/n_batches))
    # print("Testing Accuracy: ",
        #   acc(model, inputs=tf.cast(x_test, dtype="float"), targets= y_test).numpy())

print("Optimization Finished!")

# # Calculate accuracy for MNIST test images
# print("Testing Accuracy: ",
#       acc(model, inputs=tf.cast(x_test, dtype="float"), targets= y_test).numpy())