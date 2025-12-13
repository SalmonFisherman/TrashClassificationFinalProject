import tensorflow as tf
import os
from data_loader import get_datasets, get_class_weights
from model import build_model

DATA_DIR = "data/dataset/"
BATCH_SIZE = 64
EPOCHS = 15
MODEL_DIR = "output/saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
train_ds, val_ds, test_ds, class_names = get_datasets(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE
)

NUM_CLASSES = len(class_names)

# Class weight
class_weights = get_class_weights(DATA_DIR, class_names)

# Build model
model = build_model(NUM_CLASSES)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]



train_steps = tf.data.experimental.cardinality(train_ds)
val_steps = tf.data.experimental.cardinality(val_ds)

print("Train steps:", train_steps.numpy())
print("Val steps:", val_steps.numpy())

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps.numpy(),
    validation_steps=val_steps.numpy(),
    verbose=1,               
    class_weight=class_weights,
    callbacks=callbacks
)

# Save final model
model.save(os.path.join(MODEL_DIR, "final_model"))
