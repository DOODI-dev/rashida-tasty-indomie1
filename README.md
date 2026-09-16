# Chest X-Ray Pneumonia Detection using CNN

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Project Overview

This project trains an image classification model on the public **Chest X-Ray Images (Pneumonia)** dataset. The model takes a chest X-ray image as input and predicts whether the patient's lungs are normal or show signs of pneumonia, along with a confidence score.

## Project Structure

```
AI_project/
├── AI_project.ipynb          # Main Jupyter notebook (data prep, training, evaluation)
├── archive/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── new_images/                # Folder for your own X-ray images to test predictions on
├── chest_xray_model.h5        # Saved trained model
└── README.md
```

## Requirements

- Python 3.9+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow (2.21.0 or similar)
- Pillow (PIL)

Install everything with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow pillow
```

> **Note:** On native Windows, TensorFlow ≥ 2.11 does not support GPU acceleration. Training in this project runs on CPU. For GPU acceleration, use WSL2 or the TensorFlow-DirectML plugin.

## Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset (commonly sourced from Kaggle), organized into `train`, `val`, and `test` folders, each split into `NORMAL` and `PNEUMONIA` classes.

| Split | NORMAL | PNEUMONIA |
|-------|--------|-----------|
| Train | 1,341  | 3,875     |
| Test  | 234    | 390       |
| Val   | 8      | 8         |

> The dataset is **imbalanced** (more PNEUMONIA images than NORMAL) and the validation set is very small (16 images total), which affects training/evaluation reliability.

Place the extracted dataset in an `archive/chest_xray/` folder at the same level as the notebook, or update the `dataset_path` variable in the notebook to point to its location.

## How It Works (Pipeline)

1. **Verify dataset path** — confirms the dataset folder exists.
2. **Count images** — checks class balance across train/test/val.
3. **Inspect images** — visually displays sample NORMAL and PNEUMONIA X-rays, checks image dimensions and color mode.
4. **Load & preprocess data** — uses `tf.keras.utils.image_dataset_from_directory` to load images, resize them to 150×150, batch them (batch size 32), and normalize pixel values to [0, 1].
5. **Build the CNN** — a `Sequential` model with 3 convolutional blocks (Conv2D + MaxPooling2D), followed by a dense layer and a sigmoid output for binary classification.
6. **Train the model** — 10 epochs, `adam` optimizer, `binary_crossentropy` loss.
7. **Evaluate** — accuracy/loss on the held-out test set, confusion matrix, and classification report.
8. **Predict on new images** — load a single external X-ray image and classify it as NORMAL or PNEUMONIA with a confidence percentage.
9. **Save the model** — exported as `chest_xray_model.h5`.

## Model Architecture

```
Input (150, 150, 3)
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(64, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(128, 3x3, relu) → MaxPooling2D(2x2)
Flatten
Dense(128, relu)
Dense(1, sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Total parameters:** ~14.5 million

## Results

| Metric | Value |
|--------|-------|
| Final training accuracy | ~100% |
| Final validation accuracy | ~93.75% |
| **Test accuracy** | **74.68%** |
| Test loss | 4.77 |

**Classification report (test set):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.97 | 0.33 | 0.50 | 234 |
| PNEUMONIA | 0.71 | 0.99 | 0.83 | 390 |

The large gap between training accuracy (~100%) and test accuracy (~75%) indicates the model is **overfitting** — it memorized the training data rather than learning generalizable features. It also predicts PNEUMONIA far more readily than NORMAL (high recall for PNEUMONIA, low recall for NORMAL), likely due to class imbalance in the training set.

## Usage: Predicting a New X-Ray Image

```python
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("chest_xray_model.h5")

image_path = "new_images/xray1.jpeg"
img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
confidence = prediction if prediction >= 0.5 else 1 - prediction

print(f"Prediction: {label} ({confidence:.2%} confidence)")
```

## Known Issues & Recommendations for Improvement

- **Overfitting:** Add `Dropout` layers (e.g., 0.5 after the dense layer) to reduce memorization.
- **Class imbalance:** Consider class weighting (`class_weight` in `model.fit`) or oversampling the NORMAL class.
- **Small validation set:** Only 16 images in `val/` — consider re-splitting the dataset so validation gives a more reliable signal during training.
- **Data augmentation:** Use `RandomFlip`, `RandomRotation`, and `RandomZoom` to improve generalization.
- **Early stopping:** Use `EarlyStopping` callback to stop training once validation loss stops improving, instead of a fixed 10 epochs.
- **Legacy format:** The model is saved as `.h5` (legacy). Consider switching to the native Keras format (`model.save("chest_xray_model.keras")`).

## License / Data Source

This project uses the publicly available Chest X-Ray Images (Pneumonia) dataset. Please review the dataset's original license/terms of use before redistributing it.
