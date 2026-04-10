from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Central configuration for training, evaluation, and inference."""

    random_seed: int = 42
    image_size: tuple[int, int] = (32, 32)
    num_classes: int = 10
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    shuffle_before_split: bool = True   # Shuffle training data before validation split

    model_dir: Path = Path("models")
    artifacts_dir: Path = Path("artifacts")
    best_model_path: Path = Path("models/cifar10_cnn_best.keras")
    final_model_path: Path = Path("models/cifar10_cnn_final.keras")

    class_names: tuple[str, ...] = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


CONFIG = Config()
