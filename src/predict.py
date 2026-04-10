from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CONFIG
from .utils import print_section


def load_and_prepare_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and prepare it for inference."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(CONFIG.image_size)
    image_array = np.array(image, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained CIFAR-10 CNN.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--top", type=int, default=1, help="Show top K predictions")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = CONFIG.best_model_path if CONFIG.best_model_path.exists() else CONFIG.final_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model was found. Please run `python -m src.train` first."
        )

    print_section("Running inference")
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    image_array = load_and_prepare_image(image_path)
    probs = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][: args.top]

    print("\nPredictions:")
    for rank, index in enumerate(top_indices, start=1):
        label = CONFIG.class_names[index]
        confidence = float(probs[index])
        print(f"{rank}. {label}: {confidence:.4f}")


if __name__ == "__main__":
    main()
