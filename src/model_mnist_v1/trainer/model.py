"""Example implementation of code to run on the Cloud ML service.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import gzip
import sys
import _pickle as cPickle
import shutil
import glob
import re
import os
import codecs
import json
import subprocess
import requests
import google.auth
import tensorflow.contrib.rnn as rnn

print(tf.__version__)
print(tf.keras.__version__)

tf.logging.set_verbosity(tf.logging.INFO)

# learning rate
learning_rate = 0.5


# hidden layer 1
n1=300



# get mnist data, split between train and test sets
# on GCP
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# with AXA network

def load_data(path):
    f = gzip.open(path, 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
    f.close()
    return data

(x_train, y_train), (x_test, y_test) = load_data(path='../../data/mnist.pkl.gz')

# cast uint8 -> float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# renormalize the data 255 grey variation
x_train /= 255
x_test /= 255

# reshape the data 28 x 28 -> 784
x_train = x_train.reshape(len(x_train), x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(len(x_test), x_test.shape[1]*x_test.shape[2])

num_classes = len(np.unique(y_train))

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

dim_input=x_train.shape[1]


def input_dataset_fn(FLAGS, x_data, y_data, batch_size=128, mode=tf.estimator.ModeKeys.TRAIN):
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("input_dataset_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("input_dataset_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("input_dataset_fn: TRAIN, {}".format(mode))

    # 1) convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    # 2) shuffle (with a big enough buffer size)    :
    if mode == tf.estimator.ModeKeys.TRAIN:
        # num_epochs = None # loop indefinitely
        num_epochs = FLAGS.epoch
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=2)  # depends on sample size
    else:
        # num_epochs = 1 # end-of-input after this
        num_epochs = FLAGS.epoch

    print('the number of epoch: num_epoch =', num_epochs)

    # caching data
    # dataset = dataset.cache()

    # 3) automatically refill the data queue when empty
    dataset = dataset.repeat(num_epochs)

    # 4) map
    # dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_calls)

    # 5) create batches of data
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # 6) prefetch data for faster consumption, based on your system and environment, allows the tf.data runtime to automatically tune the prefetch buffer sizes
    dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)

    return dataset

# the tf.distribute.Strategy API is an easy way to distribute your training across multiple devices/machines




def baseline_model(FLAGS, opt='tf'):
    # strategy=None
    ## work with Keras with tf.train optimiser not tf.keras
    strategy = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
    # strategy = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
    # NUM_GPUS = 2
    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    # strategy = tf.contrib.distribute.MirroredStrategy()

    # config tf.estimator to use a give strategy
    training_config = tf.estimator.RunConfig(train_distribute=strategy,
                                             model_dir=FLAGS.model_dir,
                                             save_summary_steps=20,
                                             save_checkpoints_steps=20)

    # create model
    model = tf.keras.Sequential()

    # hidden layer
    model.add(tf.keras.layers.Dense(dim_input,
                                    input_dim=dim_input,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu'))
    # last layer
    model.add(tf.keras.layers.Dense(num_classes,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='softmax'))

    # weight initialisation
    # He: keras.initializers.he_normal(seed=None)
    # Xavier: keras.initializers.glorot_uniform(seed=None)
    # Radom Normal: keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # Truncated Normal: keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    if opt == 'keras':
        optimiser = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9)
        # GD/SGC:   keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # Adam:     keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # RMSProp:  keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        # Momentum: keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    else:
        # optimiser (use tf.train and not tf.keras to use MirrorStrategy)
        # https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
        optimiser = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9)
        # GD/SGC:   tf.train.GradientDescentOptimizer(learning_rate, use_locking=False, name='GradientDescent')
        # Adam:     tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam')
        # RMSProp:  tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')
        # Momentum: tf.train.MomentumOptimizer(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

    return tf.keras.estimator.model_to_estimator(keras_model=model, config=training_config)

# Create the inference model
#def simple_rnn(features, labels, mode):
#    # 0. Reformat input shape to become a sequence
#    x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)#
#
#    # 1. Configure the RNN
#    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
#    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#    # Slice to keep only the last cell of the RNN
#    outputs = outputs[-1]
#    # print('last outputs={}'.format(outputs))
#
#    # Output is result of linear activation of last layer of RNN
#    weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
#    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
#    predictions = tf.matmul(outputs, weight) + bias
#
#    # 2. Loss function, training/eval ops
#    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
#        loss = tf.losses.mean_squared_error(labels, predictions)
#        train_op = tf.contrib.layers.optimize_loss(
#            loss=loss,
#            global_step=tf.train.get_global_step(),
#            learning_rate=0.01,
#            optimizer="SGD")
#        eval_metric_ops = {
#            "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
#        }
#    else:
#        loss = None
#        train_op = None
#        eval_metric_ops = None
#
#    # 3. Create predictions
#    predictions_dict = {"predicted": predictions}
#
#    # 4. Create export outputs
#    export_outputs = {"predict_export_outputs": tf.estimator.export.PredictOutput(outputs=predictions)}
#
#    # 4. Return EstimatorSpec
#    return tf.estimator.EstimatorSpec(
#        mode=mode,
#        predictions=predictions_dict,
#        loss=loss,
#        train_op=train_op,
#        eval_metric_ops=eval_metric_ops,
#        export_outputs=export_outputs)

# Create serving input function
def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders#

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    input_images = tf.placeholder(tf.float32, [None, 784])
    features = {
        'dense_input': input_images}  # this is the dict that is then passed as "features" parameter to your model_fn
    receiver_tensors = {
        'dense_input': input_images}  # As far as I understand this is needed to map the input to a name you can retrieve later

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# Create custom estimator's train and evaluate function
def train_and_evaluate(FLAGS, use_keras):
    print('flags',FLAGS)
    if use_keras:
        estimator = baseline_model(FLAGS)
    else:
        estimator = tf.estimator.Estimator(model_fn=simple_rnn,
                                           model_dir=output_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_dataset_fn(FLAGS,
                                                                         x_train,
                                                                         y_train,
                                                                         mode=tf.estimator.ModeKeys.TRAIN,
                                                                         batch_size=FLAGS.batch_size),
                                        max_steps=1000)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_receiver_fn = serving_input_receiver_fn)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_dataset_fn(FLAGS,
                                                                       x_test,
                                                                       y_test,
                                                                       mode=tf.estimator.ModeKeys.EVAL,
                                                                       batch_size=len(x_test)),
                                      exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

