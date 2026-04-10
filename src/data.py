from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import CONFIG


def load_cifar10():
    """Load CIFAR-10 and scale pixel values to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return (x_train, y_train), (x_test, y_test)


def split_validation_data(x_train, y_train):
    """
    Create a validation split from the original training set.

    If `shuffle_before_split` is enabled, the training data is shuffled
    before the split to ensure the validation set is representative.
    """
    if CONFIG.shuffle_before_split:
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

    total_samples = x_train.shape[0]
    val_size = int(total_samples * CONFIG.validation_split)

    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]

    x_train_final = x_train[:-val_size]
    y_train_final = y_train[:-val_size]

    return (x_train_final, y_train_final), (x_val, y_val)


def make_augmentation_layer() -> tf.keras.Sequential:
    """Image augmentation used during training only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )


def build_tf_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size: int = CONFIG.batch_size):
    """Create efficient tf.data pipelines."""
    autotune = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10000, seed=CONFIG.random_seed)
        .batch(batch_size)
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(autotune)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(autotune)
    )

    return train_ds, val_ds, test_ds
