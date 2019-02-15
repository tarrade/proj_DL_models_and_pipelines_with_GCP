"""
First try to start Cloud ML

References:
Basic reference for packaging the model so that ml-engine can use it:
- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/cloudmle/taxifare
MNIST-Estimator-Example:
- https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940

ipython -i -m src.models.test_model_estimator_api.mnist_ml_engine -- --data_path=data --output_dir=src\models\test_model_estimator_api\trained --train_steps=100
"""

import tensorflow as tf
import numpy as np

from .utils import load_data
###############################################################################
#Factor into config:
N_PIXEL = 784
OUTDIR = 'trained'
USE_TPU = False
EPOCHS = 5

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
        # Boolean, if True shuffles the queue. Avoid shuffle at prediction time.
        shuffle=_shuffle,
        queue_capacity=1000,
        # Integer, number of threads used for reading and enqueueing. In order to have predicted and repeatable order of reading and enqueueing, such as in prediction and evaluation mode, num_threads should be 1.
        num_threads=_num_threads
    )


def serving_input_fn():
    feature_placeholders = {
        'x': tf.placeholder(tf.float32, shape=[None, N_PIXEL])
    }
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(
         features=features, 
         receiver_tensors=feature_placeholders,
         receiver_tensors_alternatives=None
         )


def train_and_evaluate(args):
    """
    Utility function for distributed training on ML-Engine
    https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate 
    """
    ##########################################
    # Load Data in Memoery

  # #ToDo: replace numpy-arrays
    (x_train, y_train), (x_test, y_test) = load_data(
        rel_path=args['data_path'])
  
    x_train = parse_images(x_train)
    x_test = parse_images(x_test)

    y_train = parse_labels(y_train)
    y_test = parse_labels(y_test)

    model = tf.estimator.DNNClassifier(
        hidden_units=[256, 128, 64],
        feature_columns=[tf.feature_column.numeric_column(
            'x', shape=[N_PIXEL, ])],
        model_dir=args['output_dir'],
        n_classes=10,
        optimizer=tf.train.AdamOptimizer,
        # activation_fn=,
        dropout=0.2,
        batch_norm=False,
        loss_reduction='weighted_sum',
        warm_start_from=None
    )
   
    train_spec = tf.estimator.TrainSpec(
        input_fn=numpy_input_fn(
            x_train, y_train, mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=args['train_steps']
    )
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=numpy_input_fn(
            x_test, y_test, mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=args['eval_delay_secs'],
        throttle_secs=args['min_eval_frequency'],
        exporters=exporter
    )
    tf.estimator.train_and_evaluate(
        estimator=model, train_spec=train_spec, eval_spec=eval_spec)
    print((model.get_variable_names()))




    # model.train(input_fn=numpy_input_fn(
    #     x_train, y_train, mode=tf.estimator.ModeKeys.TRAIN))
    # # #######################################

# How to evaluate in the cloud over a whole evaluation set?
    # # # Evaluate
    # metrics_train = model.evaluate(
    #     input_fn=numpy_input_fn(x_train, y_train, mode=tf.estimator.ModeKeys.EVAL))
    # metrics_test = model.evaluate(
    #     input_fn=numpy_input_fn(
    #         x_test, y_test, mode=tf.estimator.ModeKeys.EVAL)
    # )
    # import pandas as pd
    # metrics = pd.DataFrame(
    #     {'Train': metrics_train, 'Test': metrics_test}).transpose()
    # print("## Metrics DF\n", metrics)
    # # #######################################
    # # # get individual predictions:
    # predictions_iterator = model.predict(input_fn=numpy_input_fn(
    #     x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
    # for i, pred in enumerate(predictions_iterator):
    #     if i % 999 == 0:
    #         print('Image: {}'.format(i))
    #         print(pred)
    # #ToDo: 10000 Test-Images yield 20000 predictions?!
    # predictions_iterator = model.predict(input_fn=numpy_input_fn(
    #     x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
    # assert len(list(predictions_iterator)) == len(x_test)
