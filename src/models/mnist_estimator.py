import tensorflow as tf # 1.12
import numpy as np

#Factor into config:
N_PIXEL = 784

USE_TPU = False
PATH_DATA = '/../../../data/'
if USE_TPU:
    _device_update = 'tpu'
else:
    _device_update = 'cpu'

IMAGE_SIZE = 28 * 28
NUM_LABELS = 10
OUTDIR = 'trained/mnist_estimator/'
lr_inital = 0.001

BATCH_SIZE = 128
EPOCHS = 5

import shutil
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time

#######################################
# Load Data
try:
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
except Exception:
    print("download manually to ./data/ from {}".format(
      "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    ))
    with np.load("./data/mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

def parse_images(x):
    return x.reshape(len(x), -1).astype('float32')


def parse_labels(y):
    return y.astype('int32')

x_train = parse_images(x_train)
x_test = parse_images(x_test)

y_train = parse_labels(y_train)
y_test = parse_labels(y_test)

def numpy_input_fn(images: np.ndarray, 
                   labels: np.ndarray, 
                   mode=tf.estimator.ModeKeys.EVAL):
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


# Default Config:
cfg_train_default = tf.estimator.RunConfig()
# cfg_train_default.__dict__
{'_model_dir': 'trained/mnist_estimator/',
                  '_tf_random_seed': None,
                  '_save_summary_steps': 100,
                  '_save_checkpoints_steps': None,
                  '_save_checkpoints_secs': 600,
                  #   '_session_config':  allow_soft_placement: true
                  #   graph_options {
                  #       rewrite_options {
                  #           meta_optimizer_iterations: ONE
                  #       }
                  #   },
                  '_keep_checkpoint_max': 5,
                  '_keep_checkpoint_every_n_hours': 10000,
                  '_log_step_count_steps': 100,
                  '_train_distribute': None,
                  '_device_fn': None,
                  '_protocol': None, 
                  '_eval_distribute': None, 
                  '_experimental_distribute': None, 
                  '_service': None, 
                #   '_cluster_spec': < tensorflow.python.training.server_lib.ClusterSpec object at 0x7f3113ce1f60 > , 
                  '_task_type': 'worker', 
                  '_task_id': 0, 
                  '_global_id_in_cluster': 0, 
                  '_master': '', 
                  '_evaluation_master': '', 
                  '_is_chief': True, 
                  '_num_ps_replicas': 0, 
                  '_num_worker_replicas': 1}


estimator = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column(
            'x', shape=[N_PIXEL,])],
    # feature_columns: An iterable containing all the feature columns used by
    # the model. All items in the set should be instances of classes derived
    # from `_FeatureColumn`.
    hidden_units=[256, 128, 64],
    model_dir = OUTDIR, 
    n_classes = NUM_LABELS, 
    optimizer= tf.train.AdamOptimizer(), 
    # lambda: tf.train.AdamOptimizer(
        # learning_rate=tf.train.exponential_decay(
        #     learning_rate=0.001,
        #     global_step=tf.train.get_global_step(),
        #     decay_steps=10000,
        #     decay_rate=0.96
        # )),
    config = None,
    dropout= None
)

# Set up logging for predictions
# print('Variable Names:')
# print(estimator.get_variable_names())
tensors_to_log = {"probabilities": "dnn/logits/kernel/"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# def serving_input_fn():
#     """
#     Serving function which defines float32 placeholders.
#     """
#     feature_placeholders = {
#         # Note: if `features` passed is not a dict, it will be wrapped in a dict
#         #       with a single entry, using 'feature' as the key. 
#         column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
#     }
#     features = feature_placeholders
#     return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

estimator.train(input_fn=numpy_input_fn(x_train, y_train, 
                                        mode=tf.estimator.ModeKeys.TRAIN),
                # steps = 1000, 
                hooks = None
                )
# # #######################################

# How to evaluate in the cloud over a whole evaluation set?
# # Evaluate
metrics_train = estimator.evaluate(
    input_fn=numpy_input_fn(
        x_train, y_train, mode=tf.estimator.ModeKeys.EVAL))
metrics_test = estimator.evaluate(
    input_fn=numpy_input_fn(
        x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))

import pandas as pd
metrics = pd.DataFrame(
    {'Train': metrics_train, 'Test': metrics_test}).transpose()
print("## Metrics DF\n", metrics)
# #######################################
# # get individual predictions:
predictions_iterator = estimator.predict(input_fn=numpy_input_fn(
    x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
for i, pred in enumerate(predictions_iterator):
    if i % 999 == 0:
        print('Image: {}'.format(i))
        print(pred)
#ToDo: 10000 Test-Images yield 20000 predictions?!
predictions_iterator = estimator.predict(input_fn=numpy_input_fn(
    x_test, y_test, mode=tf.estimator.ModeKeys.EVAL))
assert len(list(predictions_iterator)) == len(x_test)
