# Overview Model APIs in models folder

> Tensorflow Version: 1.12
> (current: )

### `mnist_lazy.py`
- Building a graph and executing using `tf.sess`

### `mnist_eager.py` 
- Using model as in `mnist_lazy.py`, but using eager execution. 
- Eager execution is enabled by  `tf.enable_eager_execution` 

### `mnist_eager_tf_data_pipeline.py`
- same as `mnist_eager.py`using `tf.data.Dataset.from_tensor_slices`
- all reshapes and type casting done by `tf.data.Dataset`
