import os
import json
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import get_datasets
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/dataset/"
MODEL_PATH = "output/saved_model/FINAL_TEST_MODEL.keras"
OUTPUT_DIR = "output/saved_model"
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
BATCH_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

_, _, test_ds, class_names = get_datasets(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE
)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

loss, acc = model.evaluate(test_ds, verbose=1)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc:.4f}")

y_true = []
y_pred = []

for x, y in test_ds:
    preds = model.predict(x, verbose=0)
    y_true.extend(tf.argmax(y, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("loss", []), label="Training Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(OUTPUT_DIR, "loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("accuracy", []), label="Training Accuracy")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    acc_plot_path = os.path.join(OUTPUT_DIR, "accuracy_per_epoch.png")
    plt.savefig(acc_plot_path)
    plt.close()

    print(f"- Loss curve saved to {loss_plot_path}")
    print(f"- Accuracy curve saved to {acc_plot_path}")

else:
    print("- No training history found (history.json). Skipping training curves.")

print("\nEvaluation results saved:")
print(f"- {report_path}")
print(f"- {cm_path}")
