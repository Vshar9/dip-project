import tensorflow as tf
from tensorflow.keras.layers import Layer

class BlurPool(Layer):
    def call(self, inputs):
        in_channels = tf.shape(inputs)[-1]
        blur_kernel = tf.constant([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=tf.float32)
        blur_kernel = blur_kernel / 16.0
        blur_kernel = blur_kernel[:, :, tf.newaxis, tf.newaxis] 
        blur_kernel = tf.tile(blur_kernel, [1, 1, in_channels, 1])

        return tf.nn.depthwise_conv2d(inputs, blur_kernel, strides=[1, 2, 2, 1], padding="SAME")
