from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional: enable deterministic ops (may impact performance)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def save_history_plots(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Save training/validation accuracy and loss plots."""
    history_dict = history.history

    accuracy = history_dict.get("accuracy", [])
    val_accuracy = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])

    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history_dict, file, indent=2)


def print_section(title: str) -> None:
    """Pretty console separator."""
    line = "=" * len(title)
    print(f"\n{title}\n{line}")
