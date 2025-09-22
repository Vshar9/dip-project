from tensorflow.keras import layers
from .se_block import se_block
from .blur_pool import BlurPool
from .hard_swish import HardSwish

def basic_block(x, filters, stride=1):
    shortcut = x
    if stride > 1:
        x = BlurPool()(x)  # Use BlurPool class as a layer

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HardSwish()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)

    if shortcut.shape[-1] != filters or stride > 1:
        if stride > 1:
            shortcut = BlurPool()(shortcut)  # Use BlurPool class as a layer
        shortcut = layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False)(shortcut)  # Fix the typo
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = HardSwish()(x)
    return x
