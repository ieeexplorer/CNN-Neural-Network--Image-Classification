from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CONFIG
from .data import build_tf_datasets, load_cifar10, split_validation_data
from .utils import print_section


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    print_section("Evaluating model")

    model_path = CONFIG.best_model_path if CONFIG.best_model_path.exists() else CONFIG.final_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model was found. Please run `python -m src.train` first."
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_train, y_train), (x_val, y_val) = split_validation_data(x_train, y_train)

    _, _, test_ds = build_tf_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=CONFIG.batch_size
    )

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CONFIG.class_names, CONFIG.artifacts_dir / "confusion_matrix.png")
    print(f"Confusion matrix saved to {CONFIG.artifacts_dir / 'confusion_matrix.png'}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CONFIG.class_names))


if __name__ == "__main__":
    main()
