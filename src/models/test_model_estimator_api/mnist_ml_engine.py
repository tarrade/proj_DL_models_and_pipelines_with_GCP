"""
First try to start Cloud ML

References:
Basic reference for packaging the model so that ml-engine can use it:
- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/cloudmle/taxifare
MNIST-Estimator-Example:
- https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940

ipython -i -m src.models.test_model_estimator_api.mnist_ml_engine -- --train_data_path=data --output_dir=src\models\test_model_estimator_api\trained
"""
import os
import argparse
import json

from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import shutil

from src.models.test_model_estimator_api.utils import load_data

###############################################################################
#Factor into config:
N_PIXEL = 784
OUTDIR = 'trained'
USE_TPU = False
EPOCHS = 10

if USE_TPU:
    _device_update = 'tpu'
else:
    _device_update = 'cpu'

IMAGE_SIZE = 28 * 28
NUM_LABELS = 10
BATCH_SIZE = 128
###############################################################################
def parse_images(x):
    return x.reshape(len(x), -1).astype('float32')

def parse_labels(y):
    return y.astype('int32')

def numpy_input_fn(images: np.ndarray, labels: np.ndarray, mode=tf.estimator.ModeKeys.EVAL):
    if mode == tf.estimator.ModeKeys.TRAIN:
        _epochs = EPOCHS
        _shuffle = True
        _num_threads = 2
    else:
        _epochs = 1
        _shuffle = False
        _num_threads = 1

    return tf.estimator.inputs.numpy_input_fn(
        {'x': images},
        y=labels,
        batch_size=BATCH_SIZE,
        num_epochs=_epochs,
        shuffle=_shuffle, # Boolean, if True shuffles the queue. Avoid shuffle at prediction time.
        queue_capacity=1000,
        num_threads=_num_threads  # Integer, number of threads used for reading and enqueueing. In order to have predicted and repeatable order of reading and enqueueing, such as in prediction and evaluation mode, num_threads should be 1.
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_path',
        help = 'GCS or local path to training data',
        required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = '128'
    )
    parser.add_argument(
        '--hidden_units',
        help = 'List of hidden layer sizes to use for DNN feature columns',
        nargs = '+',
        type = int,
        default = [128, 64, 32]
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--job_dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )

    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = '10',
        type = int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help = 'Seconds between evaluations',
        default = 300,
        type = int
    )

    args = parser.parse_args().__dict__
    
    OUTDIR = args['output_dir']
    ##########################################
    # Load Data in Memoery
    
    #ToDo: Connect bucket:
    (x_train, y_train), (x_test, y_test) = load_data(rel_path=args['train_data_path'])
    # #ToDo: replace numpy-arrays

    x_train = parse_images(x_train)
    x_test = parse_images(x_test)

    y_train = parse_labels(y_train)
    y_test = parse_labels(y_test)

    # Define model
    model = tf.estimator.DNNClassifier(
        hidden_units=[256, 128, 64],
        feature_columns=[tf.feature_column.numeric_column('x', shape=[N_PIXEL, ])],
        model_dir=OUTDIR,
        n_classes=10,
        optimizer=tf.train.AdamOptimizer,
        # activation_fn=,
        dropout=0.2,
        batch_norm=False,
        loss_reduction='weighted_sum',
        warm_start_from=None
    )
    # #######################################
    # # Train
    shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
    model.train(input_fn=numpy_input_fn(x_train, y_train, mode=tf.estimator.ModeKeys.TRAIN))
    # #######################################
    # # Evaluate
    metrics_train = model.evaluate(
        input_fn=numpy_input_fn(x_train, y_train, mode=tf.estimator.ModeKeys.EVAL))
    metrics_test = model.evaluate(
        input_fn=numpy_input_fn(x_test, y_test, mode=tf.estimator.ModeKeys.EVAL)
    )
    import pandas as pd
    metrics = pd.DataFrame({'Train': metrics_train, 'Test': metrics_test}).transpose()
    print("## Metrics DF\n", metrics)
    # #######################################
    # # get individual predictions:
    predictions_iterator = model.predict(input_fn=numpy_input_fn(x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
    for i, pred in enumerate(predictions_iterator):
        if i % 999 == 0:
            print('Image: {}'.format(i))
            print(pred)
    #ToDo: 10000 Test-Images yield 20000 predictions?!
    predictions_iterator = model.predict(input_fn=numpy_input_fn(x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
    assert len(list(predictions_iterator)) == len(x_test)
