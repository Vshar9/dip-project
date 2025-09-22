import tensorflow as tf
from tensorflow.keras import layers, models

class HardSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.nn.relu6(inputs + 3) / 6