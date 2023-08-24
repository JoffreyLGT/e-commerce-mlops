"""Check if GPU is supported with Tensorflow."""


import tensorflow as tf


gpus = tf.config.list_physical_devices("GPU")
print(f"GPU support: {len(gpus) > 0}")
print(f"GPU list: {gpus}")
