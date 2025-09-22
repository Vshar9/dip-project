from tensorflow.keras import layers

def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape([1,1,filters])(se)
    return layers.multiply([x,se])