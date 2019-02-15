"""Example implementation of code to run on the Cloud ML service.
"""

import traceback
import argparse
import json
import os

from . import model

import shutil
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
from absl import app
#from absl import flags

#def main(_):
#  with logger.benchmark_context(flags.FLAGS):
#    run_cifar(flags.FLAGS)

# number of epoch to train our model
EPOCHS = 2

# size of our mini batch
#BATCH_SIZE = len(x_train)
BATCH_SIZE = 128

# shuffle buffer size
SHUFFLE_BUFFER_SIZE = 10 * BATCH_SIZE

# prefetch buffer size
PREFETCH_BUFFER_SIZE = 1 #tf.contrib.data.AUTOTUNE

# number of paralell calls
NUM_PARALELL_CALL = 4

flags.DEFINE_string('model_dir', './results/Models/Mnist/tf_1_12/estimator/ckpt/',
                    'Dir to save a model and checkpoints')
flags.DEFINE_string('saved_dir', './results/Models/Mnist/tf_1_12/estimator/pt/',
                    'Dir to save a model for TF serving')
flags.DEFINE_integer('shuffle_buffer_size', SHUFFLE_BUFFER_SIZE, 'Shuffle buffer size')
flags.DEFINE_integer('prefetch_buffer_size', PREFETCH_BUFFER_SIZE, 'Prefetch buffer size')
flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size')
flags.DEFINE_integer('epoch', EPOCHS, 'number of epoch')
flags.DEFINE_integer('num_parallel_calls', NUM_PARALELL_CALL, 'Number of paralell calls')

def main(args):

    # Run the training job
    try:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)  # start fresh each time
        model.train_and_evaluate(FLAGS, True)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    #define_cifar_flags()
    #app.run(main)
    #traceback.print_exc()
    #with logger.benchmark_context(flags.FLAGS):

    tf.app.run()