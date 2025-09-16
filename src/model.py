"""
model.py
Contains MLP and fusion (CNN + dense) model constructors.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def create_mlp(input_dim: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_cnn_dense_fusion(img_shape=(128,128,3), num_features=3):
    # image branch
    img_input = layers.Input(shape=img_shape, name="img_input")
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(img_input)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # numerical branch
    num_input = layers.Input(shape=(num_features,), name="num_input")
    y = layers.Dense(64, activation='relu')(num_input)
    y = layers.Dense(32, activation='relu')(y)

    # fusion
    z = layers.concatenate([x,y])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation='sigmoid')(z)

    model = models.Model(inputs=[img_input, num_input], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
