"""
    Example implementation of code to run on the Cloud ML service.
"""

import traceback
import shutil
import tensorflow as tf
from . import model

flags = tf.app.flags
FLAGS = flags.FLAGS

# number of classes
NUM_CLASSES =10

# dimension of the input data
DIM_INPUT = 784

# number of epoch to train our model
EPOCHS = 100

# size of our mini batch
BATCH_SIZE = 128

# shuffle buffer size
SHUFFLE_BUFFER_SIZE = 10 * BATCH_SIZE

# prefetch buffer size
PREFETCH_BUFFER_SIZE = tf.contrib.data.AUTOTUNE

# number of paralell calls
NUM_PARALELL_CALL = 4

# path to input data
flags.DEFINE_string('input_train_tfrecords', '', 'input folder of tfrecords training data')
flags.DEFINE_string('input_test_tfrecords', '', 'input folder of tfrecords testing data')

# path to store the model and input for Tensorboard and SavedModel
flags.DEFINE_string('model_dir', 'results/Models/Mnist/tf_1_12/estimator/ckpt/', 'Dir to save a model and checkpoints')
flags.DEFINE_string('saved_dir', 'results/Models/Mnist/tf_1_12/estimator/pt/', 'Dir to save a model for TF serving')

# parameters for the input dataset and train the model
flags.DEFINE_integer('epoch', EPOCHS, 'number of epoch')
flags.DEFINE_integer('step_per_epoch', 1, 'number of step per epoch')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size')
flags.DEFINE_integer('shuffle_buffer_size', SHUFFLE_BUFFER_SIZE , 'Shuffle buffer size')
flags.DEFINE_integer('prefetch_buffer_size', PREFETCH_BUFFER_SIZE, 'Prefetch buffer size')
flags.DEFINE_integer('num_parallel_calls', NUM_PARALELL_CALL, 'Number of paralell calls')

# parameters for the model
flags.DEFINE_integer('num_classes', NUM_CLASSES, 'number of classes in our model')
flags.DEFINE_integer('dim_input', DIM_INPUT, 'dimension of the input data for our model')


def main(args):

    # run the training job
    try:
        # clean the directory before retraining the model
        # start fresh each time
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
        shutil.rmtree(FLAGS.saved_dir, ignore_errors=True)

        # reset Keras
        tf.keras.backend.clear_session()

        # use the estimator model with keras layer per default
        model.train_and_evaluate(FLAGS, False)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    tf.app.run()