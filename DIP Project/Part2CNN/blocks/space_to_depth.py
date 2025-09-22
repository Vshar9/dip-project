import tensorflow as tf
from tensorflow.keras import layers, models

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

def stem_block(inputs):
    x = layers.Lambda(space_to_depth_x2)(inputs)
    x = layers.Conv2D(64,3,padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    from .hard_swish import HardSwish
    x = HardSwish()(x)
    return x
