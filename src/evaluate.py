import tensorflow as tf
import os
from data_loader import get_datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_DIR = "data/dataset/"
MODEL_PATH = "output/saved_model/new_final_model_no_more.keras"
OUTPUT_DIR = "output/saved_model"
BATCH_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
_, _, test_ds, class_names = get_datasets(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile = False)

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Evaluate
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc:.4f}")

# Predictions
y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x, verbose=0)
    y_true.extend(tf.argmax(y, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

# Generate reports
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

cm = confusion_matrix(y_true, y_pred)

# Save Classification Report
report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)

# Save Confusion Matrix
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.txt")
with open(cm_path, "w") as f:
    f.write("Confusion Matrix\n")
    f.write("================\n\n")
    f.write(np.array2string(cm))

print("\nEvaluation results saved:")
print(f"- {report_path}")
print(f"- {cm_path}")
