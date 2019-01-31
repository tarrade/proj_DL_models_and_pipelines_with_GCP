import tensorflow as tf
from utils import load_data

#Factor into config:
N_PIXEL = 784

USE_TPU = False

if USE_TPU:
    _device_update = 'tpu'
else:
    _device_update = 'cpu'

IMAGE_SIZE = 28 * 28
NUM_LABELS = 10

#ToDo: Connect bucket:
(x_train, y_train), (x_test, y_test) = load_data(rel_path='/../../../data/')

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

def serving_input_fn():

    feature_placeholders = {
        # Note: if `features` passed is not a dict, it will be wrapped in a dict
        #       with a single entry, using 'feature' as the key. 
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# tf.estimator.train_and_evaluate

# def make_input_fn(df, num_epochs):
#   return tf.estimator.inputs.pandas_input_fn(
#     x = df,
#     y = df[LABEL],
#     batch_size = 128,
#     num_epochs = num_epochs,
#     shuffle = True,
#     queue_capacity = 1000,
#     num_threads = 1
#   )

# Create an estimator that we are going to train and evaluate
def train_and_evaluate(args):
    estimator = tf.estimator.DNNRegressor(
        model_dir = args['output_dir'],
        feature_columns = feature_cols,
        hidden_units = args['hidden_units'])
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args['train_data_paths'],
                                batch_size = args['train_batch_size'],
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(args['eval_data_paths'],
                                batch_size = 10000,
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        start_delay_secs = args['eval_delay_secs'],
        throttle_secs = args['min_eval_frequency'],
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def fit_batch(img, labels):
  with tf.variable_scope('cpu', reuse=tf.AUTO_REUSE): # gpu, tpu
    # flatten images
    x = tf.reshape(img, [-1, IMAGE_SIZE])
    
    #MODEL:
    W = tf.get_variable('W', [28*28, 10])  # pylint: disable=invalid-name
    b = tf.get_variable('b', [10], initializer=tf.zeros_initializer)
    logits = tf.matmul(x, W) + b
    print(img, logits, labels)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    # optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer) #TPU specific

    return loss, optimizer.minimize(loss, tf.train.get_or_create_global_step())


images = tf.placeholder(name='images', dtype=tf.float32, shape=[None, 28, 28])
labels = tf.placeholder(name='labels', dtype=tf.int32, shape=[None,])

fit_on_tpu = fit_batch(images, labels)
# fit_on_tpu = tf.contrib.tpu.rewrite(fit_batch, [images, labels]) # TPU specific

session.run(tf.global_variables_initializer())
for i in range(50):
  loss = session.run(fit_on_tpu, {
      images: x_train[:1000], labels: y_train[:1000]
  })
  if i % 10 == 0:
    print('loss = %s' % loss)

def predict(img):
  with tf.variable_scope('tpu', reuse=tf.AUTO_REUSE):
    # flatten images
    x = tf.reshape(img, [-1, IMAGE_SIZE])
    
    W = tf.get_variable('W', [28*28, 10])  # pylint: disable=invalid-name
    b = tf.get_variable('b', [10], initializer=tf.zeros_initializer)
    logits = tf.matmul(x, W) + b
    return tf.nn.softmax(logits)

predict_on_tpu = tf.contrib.tpu.rewrite(predict, [images,])

from matplotlib import pyplot
%matplotlib inline

def plot_predictions(images, predictions):
  f, axes = pyplot.subplots(16, 2)
  for i in range(16):
    axes[i, 0].bar(np.arange(10), predictions[i])
    axes[i, 1].imshow(images[i])
    axes[i, 1].axis('off')

    if i != 15:
      axes[i, 0].axis('off')
    else:
      axes[i, 0].get_yaxis().set_visible(False)
  pyplot.gcf().set_size_inches(6, 8)  


[predictions] = session.run(predict_on_tpu, {
    images: x_test[:16],
})
plot_predictions(x_test[:16], predictions)