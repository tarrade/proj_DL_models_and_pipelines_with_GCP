# Overview Model APIs in models folder

> Tensorflow Version: 1.12

- all *scripts* have to be packaged in order to run with `gcloud ml-engine`.

framework_tpyes

- types might be combinations

framework: tf, keras
types tf: estimator, lowlevel, eager, lazy
types keras: cnn

- keras is always eager?


### `tf_lazy_lowlevel.py`
- Building a graph and executing using `tf.sess`

### `tf_cnn_lowlevel`
- ToDo

### `keras_eager_Dataset.py` 
- Using model as in `mnist_lazy.py`, but using eager execution. 
- Eager execution is enabled by  `tf.enable_eager_execution` 

### `tf_estimator_fnn`
- Includes Estimator-API using
   - `tf.Dataset` using `tf.estimator.inputs.numpy_input_fn`
   - `tf.estimator.RunConfig`
   - `tf.estimator.DNNClassifier`
   - `tf.train.LoggingTensorHook`

### `tf_estimator_cnn.py`
- ToDo

### `keras_cnn_eager`
- ToDo

### `keras_eager_loss_grad_example`
- ToDo
- not MNIST

### `tf_distributed_testskript`
- ToDo
- not MNIST

### `keras_eager_multigpu`
- ToDo: to test and describe


### `keras_eager_tf_data.py`
- same as `mnist_eager.py`using `tf.data.Dataset.from_tensor_slices`
- all reshapes and type casting done by `tf.data.Dataset`

### Reload a model from Checkpoints
- Test: Estimator load `model_dir` parameter of `tf.estimator.Estimator`
- Test: `tf.saved_model.load`


### to adapt (maybe only load Modelling part)
- `mnist_tf_guide_cnn_eager.py`
- `tf_distributed_testskript.py`
- `tf_eager_loss_grad_example.py`
- `mnist_eager_multigpu.py`
- `cnn_mnist_tf_example.py`
