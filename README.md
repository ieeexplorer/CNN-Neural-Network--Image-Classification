# CNN Neural Network for Image Classification (CIFAR-10)

A complete **GitHub-ready TensorFlow project** for training, evaluating, and using a **Convolutional Neural Network (CNN)** on the **CIFAR-10** image classification dataset.

This project demonstrates a practical deep learning workflow using:

- **TensorFlow / Keras**
- **Python**
- **Data augmentation** (rotation, zoom, horizontal flip)
- **Dropout regularisation**
- **Model checkpoints**
- **Early stopping**
- **Learning-rate scheduling**
- **Training history plots**
- **Reusable inference script**

The dataset contains **60,000 colour images** across **10 classes**:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## Why this project is useful

This is the kind of project that shows employers you can:

- build a real deep learning pipeline
- structure a project like production code
- handle image preprocessing and augmentation
- reduce overfitting with regularisation
- evaluate model performance properly
- package work cleanly for GitHub and a CV

---

## Expected outcome

With a reasonable training run and standard hardware, this architecture is designed to achieve **strong CIFAR-10 performance**.  
A well-tuned run can typically reach **around 80–85% validation accuracy**, depending on environment, random seed, and training duration.

> Note: this repository is provided as a ready-to-run project template. The exact final score depends on your machine and training run.

---

## Project structure

```text
cnn-cifar10-image-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── cv_bullets.md
├── notebooks/
│   └── starter_exploration.ipynb
├── artifacts/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── utils.py
```

---

## Installation

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to train

Run from the project root:

```bash
python -m src.train
```

This will:

- download CIFAR-10 automatically
- preprocess the data
- train the CNN
- save the best model into `models/`
- save loss/accuracy plots into `artifacts/`

---

## How to evaluate

```bash
python -m src.evaluate
```

This will load the saved model and print test loss and accuracy.

---

## How to predict on your own image

```bash
python -m src.predict --image path/to/your_image.jpg
```

You can also show multiple predictions with `--top`:
```bash
python -m src.predict --image path/to/your_image.jpg --top 3
```

The script will:

- resize the image to `32x32`
- normalise pixel values
- run inference
- print the predicted class and confidence

You can also visualize sample predictions with:
```bash
python -m src.visualize --num_samples 16 --use_test
```

---

## Model design

The network uses:

- stacked convolutional blocks
- batch normalisation
- max pooling
- dropout
- dense classifier head with softmax output

Regularisation strategy includes:

- image augmentation
- dropout
- early stopping
- learning-rate reduction on plateau

---

## Example CV bullets

See: [`docs/cv_bullets.md`](docs/cv_bullets.md)

Short version:

- Built and trained a CNN in TensorFlow/Python for CIFAR-10 image classification across 10 classes and 60,000 colour images.
- Applied data augmentation (rotation, zoom, horizontal flip) and dropout regularisation to reduce overfitting and improve generalisation.
- Developed an end-to-end workflow covering preprocessing, model training, evaluation, checkpointing, and inference.
- Achieved strong validation performance, targeting approximately 80–85% accuracy depending on training configuration and runtime environment.

---

## Fast learning explanation

| Part | What it means in simple words |
|---|---|
| CIFAR-10 | A famous image dataset used to train image classifiers |
| CNN | A neural network that is especially good at finding patterns in images |
| Data augmentation | Slightly changes training images so the model learns better |
| Dropout | Randomly turns off some neurons during training so the model does not memorise too much |
| Validation accuracy | How well the model performs on unseen validation data |
| Inference | Using the trained model to predict a new image |

---

## Tips and tricks

| Tip | Why it matters |
|---|---|
| Keep augmentation moderate | Too much augmentation can make training unstable |
| Use dropout after deeper layers | Helps reduce overfitting without destroying early visual features |
| Save the best checkpoint | Prevents losing your strongest model |
| Plot training history | Lets you spot overfitting quickly |
| Add early stopping | Avoids wasting time once validation stops improving |
| Start simple, then improve | Easier to debug than a very complex model |

---

## Suggested GitHub repo title

`cnn-cifar10-image-classification`

## Suggested GitHub description

TensorFlow CNN for CIFAR-10 image classification with augmentation, dropout, evaluation, and inference pipeline.

---

## Next upgrades

If you want to extend this project later, good improvements are:

- confusion matrix and per-class accuracy
- transfer learning with a stronger backbone
- TensorBoard logging
- Docker support
- FastAPI inference API
- experiment tracking with MLflow

---***---
