from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .config import CONFIG
from .data import load_cifar10, split_validation_data
from .utils import print_section


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sample predictions.")
    parser.add_argument("--num_samples", type=int, default=9, help="Number of images to display")
    parser.add_argument("--use_test", action="store_true", help="Use test set instead of validation")
    args = parser.parse_args()

    print_section("Visualizing predictions")

    model_path = CONFIG.best_model_path if CONFIG.best_model_path.exists() else CONFIG.final_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model was found. Please run `python -m src.train` first."
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    if args.use_test:
        x_data, y_data = x_test, y_test
        dataset_name = "test"
    else:
        (_, _), (x_data, y_data) = split_validation_data(x_train, y_train)
        dataset_name = "validation"

    indices = np.random.choice(len(x_data), size=min(args.num_samples, len(x_data)), replace=False)
    images = x_data[indices]
    true_labels = y_data[indices]

    probs = model.predict(images, verbose=0)
    pred_labels = np.argmax(probs, axis=1)

    cols = int(np.ceil(np.sqrt(len(indices))))
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(len(indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        true_name = CONFIG.class_names[true_labels[i]]
        pred_name = CONFIG.class_names[pred_labels[i]]
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color)
        plt.axis("off")

    plt.tight_layout()
    save_path = CONFIG.artifacts_dir / f"sample_predictions_{dataset_name}.png"
    plt.savefig(save_path)
    print(f"Sample predictions saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
