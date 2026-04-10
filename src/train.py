from __future__ import annotations

import argparse
import tensorflow as tf

from .config import CONFIG
from .data import build_tf_datasets, load_cifar10, split_validation_data
from .model import build_cnn_model
from .utils import ensure_dirs, print_section, save_history_plots, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate, help="Learning rate")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_section("Starting training")
    set_seed(CONFIG.random_seed)
    ensure_dirs(CONFIG.model_dir, CONFIG.artifacts_dir)

    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_train, y_train), (x_val, y_val) = split_validation_data(x_train, y_train)

    train_ds, val_ds, _ = build_tf_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=args.batch_size
    )

    print("Building model...")
    model = build_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CONFIG.best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if not args.no_early_stop:
        callbacks.insert(
            1,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
        )

    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Saving final model to: {CONFIG.final_model_path}")
    model.save(CONFIG.final_model_path)

    print("Saving plots and history...")
    save_history_plots(history, CONFIG.artifacts_dir)

    print_section("Training complete")
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
