from tensorflow.keras import layers, models
from blocks.space_to_depth import stem_block
from blocks.stage_block import stage_block

def build_tresnet_cifar10(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = stem_block(inputs)
    x = stage_block(x, filters=64, num_blocks=3, downsample=1)
    x = stage_block(x, filters=128, num_blocks=3, downsample=2)
    x = stage_block(x, filters=256, num_blocks=3, downsample=2)
    x = stage_block(x, filters=512, num_blocks=2, downsample=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
