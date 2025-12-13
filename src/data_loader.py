import tensorflow as tf
import os
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def get_datasets(
    data_dir,
    img_size=(224, 224), # size udah 224x224
    batch_size=64,
    val_split=0.2,
    test_split=0.1,
    seed=13
):
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    class_names = full_ds.class_names

    total_batches = tf.data.experimental.cardinality(full_ds).numpy()

    test_batches = int(total_batches * test_split)
    val_batches = int(total_batches * val_split)

    test_ds = full_ds.take(test_batches)
    val_ds = full_ds.skip(test_batches).take(val_batches)
    train_ds = full_ds.skip(test_batches + val_batches)

    # normalization layer
    aug = get_augmentation()

    # preprocess langsung pakai mobilenet_v2 preprocess_input untuk mempermudah
    # karena sudah termauk normalisasi yang diperlukan oleh model.
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(aug(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def get_class_weights(data_dir, class_names):
    labels = []
    for folder in class_names:
        path = os.path.join(data_dir, folder)
        if os.path.exists(path):
            count = len(os.listdir(path))
            labels.extend([class_names.index(folder)] * count)
    
    if not labels:
        return None

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(weights))