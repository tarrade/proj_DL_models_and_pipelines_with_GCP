# MNIST example from eager execution guide 
# https://www.tensorflow.org/guide/eager#train_a_model
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

BATCH = 32
# Fetch and format the mnist data
# (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
try:
    raise Exception
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (X_test, y_test) = mnist.load_data()
except Exception:
  print("download manually to ./data/ from {}".format(
      "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
  ))
  with np.load("./data/mnist.npz") as f:
    mnist_images, mnist_labels = f['x_train'], f['y_train']
    test_images, test_labels = f['x_test'], f['y_test']

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(BATCH)

# Build the model
# ToDo: replace by lower api model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])


for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())
  
optimizer = tf.train.AdamOptimizer()

loss_history = []

i = 1
N_train = len(mnist_images)
n_batches = N_train / BATCH
step = int(n_batches/ 60)
print("Epoch {}: ".format(i), end='')
for (batch, (images, labels)) in enumerate(dataset.take(-1)):
  if batch % step == 0:
    print('#', end='')
  with tf.GradientTape() as tape: # Provide Foward-Path for Gradient calc
    logits = mnist_model(images, training=True)
    loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  loss_history.append(loss_value.numpy())
  grads = tape.gradient(loss_value, mnist_model.variables)  # tape instance is closed after being called once
  optimizer.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()