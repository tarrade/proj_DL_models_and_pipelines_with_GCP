"""Example implementation of code to run on the Cloud ML service.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.misc import imread
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

#print(tf.__version__)
#print(tf.keras.__version__)

tf.logging.set_verbosity(tf.logging.DEBUG)

# learning rate
#learning_rate = 0.5


# hidden layer 1
#n1=300



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

#(x_train, y_train), (x_test, y_test) = load_data(path='../../data/mnist.pkl.gz')

# cast uint8 -> float32
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

# renormalize the data 255 grey variation
#x_train /= 255
#x_test /= 255

# reshape the data 28 x 28 -> 784
#x_train = x_train.reshape(len(x_train), x_train.shape[1]*x_train.shape[2])
#x_test = x_test.reshape(len(x_test), x_test.shape[1]*x_test.shape[2])

#num_classes = len(np.unique(y_train))

# convert class vectors to binary class matrices
#y_train = tf.keras.utils.to_categorical(y_train, num_classes)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#dim_input=x_train.shape[1]


def mnist_preprocessing_fn(image, label, FLAGS):
    # reshape the images from 28 x 28 to 784
    image = tf.reshape(image, [FLAGS.dim_input])

    # cast images from uint8 to float32
    image = tf.cast(image, tf.float32)

    # renormalize images 255 grey variation
    image /= 255

    # convert class vectors to binary class matrices
    label = tf.one_hot(label, FLAGS.num_classes)

    return {'dense_input':image}, label


def input_mnist_array_dataset_fn(x_data, y_data, FLAGS, batch_size=128, mode=tf.estimator.ModeKeys.TRAIN):
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
        #num_epochs = 1 # end-of-input after this -> bug in keras or feature? https://github.com/tensorflow/tensorflow/issues/25254#issuecomment-459824771
        num_epochs = FLAGS.epoch

    # 3) automatically refill the data queue when empty
    dataset = dataset.repeat(num_epochs)

    # 4) map
    dataset = dataset.map(lambda x, y: mnist_preprocessing_fn(x, y, FLAGS), num_parallel_calls=FLAGS.num_parallel_calls)

    # 5) create batches of data
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # 6) prefetch data for faster consumption, based on your system and environment, allows the tf.data runtime to automatically tune the prefetch buffer sizes
    dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)

    return dataset


def _int64_feature(value: int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value: str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature

    Args:
        value: The value to store in the feature

    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _data_path(data_directory: str, name: str) -> str:
    """Construct a full path to a TFRecord file to be stored in the
    data_directory. Will also ensure the data directory exists

    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord

    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, f'{name}.tfrecords')

def _numpy_to_tfrecords(example_dataset, filename:str):
    print(f'Processing {filename} data')
    dataset_length = len(example_dataset)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index, (image, label) in enumerate(example_dataset):
            sys.stdout.write(f"\rProcessing sample {index+1} of {dataset_length}")
            sys.stdout.flush()
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        print()


def convert_numpy_to_tfrecords(x_data, y_data, name: str, data_directory: str, num_shards: int = 1):
    """Convert the dataset into TFRecords on disk

    Args:
        x_data:         The MNIST data set to convert: data
        y_data:         The MNIST data set to convert: label
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """

    data_set = list(zip(x_data, y_data))
    data_directory = os.path.abspath(data_directory)

    if num_shards == 1:
        _numpy_to_tfrecords(data_set, _data_path(data_directory, name))
    else:
        sharded_dataset = np.array_split(data_set, num_shards)
        for shard, dataset in enumerate(sharded_dataset):
            _numpy_to_tfrecords(dataset, _data_path(data_directory, f'{name}-{shard + 1}'))

def _image_to_tfrecords(image_paths, filename:str):
    print(f'Processing {filename} data')
    length = len(image_paths)
    #print(image_paths.split('_')[-1])
    #labels=int(image_paths.split('_')[-1])
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index, image_path in enumerate(image_paths):
            label = int(''.join([n for n in image_path.split('_')[-1] if n.isdigit()]))
            sys.stdout.write(f"\rProcessing sample {index+1} of {length}")
            sys.stdout.flush()

            # Load the image-file using matplotlib's imread function.
            image = imread(image_path)

            # Convert the image to raw bytes
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        print()

def convert_image_to_tfrecords(input_images, name: str, data_directory: str, num_shards: int = 1):
    """Convert the dataset into TFRecords on disk

    Args:
        path:           The MNIST data set location
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """

    data_directory = os.path.abspath(data_directory)

    if num_shards == 1:
        _image_to_tfrecords(input_images, _data_path(data_directory, name))
    else:
        sharded_dataset = np.array_split(input_images, num_shards)
        for shard, dataset in enumerate(sharded_dataset):
            _image_to_tfrecords(dataset, _data_path(data_directory, f'{name}-{shard + 1}'))


def input_mnist_tfrecord_dataset_fn(filenames, FLAGS, batch_size=128, mode=tf.estimator.ModeKeys.TRAIN):

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("input_dataset_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("input_dataset_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("input_dataset_fn: TRAIN, {}".format(mode))

    def _parser(record):

        # 1. define a parser
        features = {
            # the label are parsed as int
            'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
            # the bytes_list data is parsed into tf.string.
            'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
        parsed_record = tf.parse_single_example(record, features)

        # 2. Convert the data
        label = parsed_record['label']
        image = tf.cast(tf.decode_raw(parsed_record['image_raw'], out_type=tf.uint8), tf.float64)

        return image, label

    def _input_fn():

        drop_remainder = True

        # 1) creating a list of files
        file_list = tf.gfile.Glob(filenames)

        # 2) read data from TFRecordDataset
        dataset = (tf.data.TFRecordDataset(file_list).map(_parser))

        # 3) shuffle (with a big enough buffer size)
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = FLAGS.epoch  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, seed=2)  # depends on sample size
        else:
            num_epochs = 1 # end-of-input after this -> bug in keras or feature? https://github.com/tensorflow/tensorflow/issues/25254#issuecomment-459824771
            #num_epochs = FLAGS.epoch
            drop_remainder = False
            print('not training', num_epochs, FLAGS.epoch)

        # 4) automatically refill the data queue when empty
        dataset = dataset.repeat(num_epochs)

        # 5) map
        dataset = dataset.map(lambda x, y: mnist_preprocessing_fn(x, y, FLAGS),
                              num_parallel_calls=FLAGS.num_parallel_calls)

        # 6) create batches of data
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        # 7) prefetch data for faster consumption, based on your system and environment, allows the tf.data runtime to automatically tune the prefetch buffer sizes
        dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)

        return dataset


    return _input_fn()

# old to be drop when all the rest is working
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


# creating the layer of a model with keras
def keras_building_blocks(dim_input, num_classes):

    # create model
    model = tf.keras.Sequential()

    # hidden layer
    model.add(tf.keras.layers.Dense(512,
                                    input_dim=dim_input,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(512,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # last layer
    model.add(tf.keras.layers.Dense(num_classes,
                                    kernel_initializer=tf.keras.initializers.he_normal(),
                                    bias_initializer=tf.keras.initializers.Zeros(),
                                    activation='softmax'))

    return model


# building a full keras model
def keras_baseline_model(dim_input, num_classes, opt='tf'):

    # gettings the bulding blocks
    model = keras_building_blocks(dim_input, num_classes)

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
    return model


# convert a keras model to an estimator model
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

    model = keras_baseline_model(FLAGS.dim_input, FLAGS.num_classes,opt='tf')

    return tf.keras.estimator.model_to_estimator(keras_model=model, config=training_config)


# estimator model
def baseline_estimator_model(features, labels, mode, params):
    """
    Model function for Estimator
    """
    # Build the model using keras layers
    # should we put   model(image, training=False) for predict
    # or should weset the learning phase
    if mode == tf.estimator.ModeKeys.TRAIN:
        K.set_learning_phase(True)
    else:
        K.set_learning_phase(False)

    # gettings the bulding blocks
    model = keras_building_blocks(params['dim_input'], params['num_classes'])

    dense_inpout = features['dense_input']

    # Logits layer
    logits = model(dense_inpout)

    # Compute predictions
    probabilities = tf.nn.softmax(logits)
    classes = tf.argmax(probabilities, axis=1, )

    # made prediction
    predictions = {
        'classes': classes,
        'probabilities': probabilities,
    }

    # to be tested
    predictions_output = tf.estimator.export.PredictOutput(predictions)

    #predictions_output = {
    #    'classify': tf.estimator.export.PredictOutput(predictions)
    #}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predictions_output})

    # Compute loss for both TRAIN and EVAL modes
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Generate necessary evaluation metrics
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=classes, name='accuracy')
    eval_metrics = {'accuracy': accuracy}

    tf.summary.scalar('accuracy', accuracy[1])

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metrics)

    # Provide an estimator spec for `ModeKeys.TRAIN`
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_or_create_global_step())


        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

        #return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=evalmetrics, export_outputs=predictions_output)
        #return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, export_outputs=predictions_output)


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
        'dense_input4': input_images}  # As far as I understand this is needed to map the input to a name you can retrieve later

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# Create custom estimator's train and evaluate function
def train_and_evaluate(FLAGS, use_keras=True):

    tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file
    EVAL_INTERVAL = 10  # seconds



    if use_keras:
        estimator = baseline_model(FLAGS)
    else:
        strategy = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
        run_config = tf.estimator.RunConfig(train_distribute=strategy,
                                            model_dir=FLAGS.model_dir,
                                            save_summary_steps=10,      # save summary every n steps
                                            save_checkpoints_steps=10,  # save model every iteration (needed for eval)
                                            #save_checkpoints_secs=10,
                                            keep_checkpoint_max=3,      # keep last n models
                                            log_step_count_steps=50)    # global steps in log and summary

        estimator = tf.estimator.Estimator(model_fn=baseline_estimator_model,
                                           params={'dim_input': FLAGS.dim_input, 'num_classes': FLAGS.num_classes},
                                           config=run_config,
                                           model_dir=FLAGS.model_dir)

    # training
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_mnist_tfrecord_dataset_fn(FLAGS.input_train_tfrecords,
                                                                                        FLAGS,
                                                                                        mode=tf.estimator.ModeKeys.TRAIN,
                                                                                        batch_size=FLAGS.batch_size),
                                        max_steps=1000)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_receiver_fn = serving_input_receiver_fn)

    print('exporter',exporter)

    # evaluation
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_mnist_tfrecord_dataset_fn(FLAGS.input_test_tfrecords,
                                                                                      FLAGS,
                                                                                      mode=tf.estimator.ModeKeys.EVAL,
                                                                                      batch_size=10000),
                                      steps=1,
                                      start_delay_secs=0,
                                      throttle_secs=0,
                                      exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def train_and_evaluate_old(FLAGS, use_keras):

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

