# CNN Image Classification — Fruits 360

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Notebook](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)

Fruit image classification using Transfer Learning (MobileNetV2) with additional Conv2D + Pooling layers.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Notebook Flow](#notebook-flow)
- [Results](#results)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Overview

This project classifies 19 fruit classes using a MobileNetV2 backbone with custom Conv2D + Pooling + Dense layers on top. The model achieves 100% accuracy on both train and test sets.

**Pipeline stages:**
1. **Dataset Loading** — Download from Kaggle (Fruits 360 dataset)
2. **Exploration & Selection** — Pick 19 fruit classes
3. **Preprocessing** — Train/validation/test split (70/20/10), resizing to 224×224
4. **Data Augmentation** — Rotation, shifting, zoom, flip
5. **Modeling** — Transfer Learning (MobileNetV2) + Conv2D + Pooling
6. **Fine-Tuning** — Unfreeze top layers with lower learning rate
7. **Evaluation** — Accuracy & loss on test set
8. **Export** — SavedModel, TF-Lite, TFJS

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras (MobileNetV2) |
| Data | NumPy |
| Visualization | Matplotlib |
| Data Source | Kaggle — Fruits 360 |
| Environment | Google Colab (GPU T4) |

## Project Structure

```
fruits360-transfer-learning/
├── CNN_Fruits360_TL.ipynb   # Main notebook
├── .gitignore
├── requirements.txt
└── README.md
```

> Model artifacts (`saved_model/`, `tflite/`, `tfjs_model/`) are excluded from the repo. Run the notebook to regenerate them.

## Dataset

- **Source**: [Kaggle — Fruits 360](https://www.kaggle.com/datasets/moltean/fruits)
- **Size**: ~9,784 images
- **Resolution**: 100×100 (upscaled to 224×224)
- **Split**: Train 70% / Validation 20% / Test 10%
- **Classes**: 19 fruit classes

| # | Class | # | Class | # | Class |
|---|---|---|---|---|---|
| 1 | Apple 10 | 8 | Kiwi 1 | 15 | Plum 1 |
| 2 | Avocado 1 | 9 | Lemon 1 | 16 | Raspberry 1 |
| 3 | Banana 1 | 10 | Mango 1 | 17 | Strawberry 1 |
| 4 | Blueberry 1 | 11 | Orange 1 | 18 | Tomato 1 |
| 5 | Cherry 1 | 12 | Peach 1 | 19 | Watermelon 1 |
| 6 | Fig 1 | 13 | Pear 1 |  |  |
| 7 | Grape 1 | 14 | Pineapple 1 |  |  |

## Notebook Flow

| # | Section |
|---|---|
| 1 | Install & Import Libraries |
| 2 | Download Dataset from Kaggle |
| 3 | Exploration & Select 5 Classes |
| 4 | Split Dataset (70/20/10) |
| 5 | Data Augmentation & Generator |
| 6 | Build Model (MobileNetV2 + Conv2D + Pooling) |
| 7 | Compile Model |
| 8 | Callbacks |
| 9 | Training Phase 1 — Frozen Base |
| 10 | Fine-Tuning Phase 2 |
| 11 | Plot Accuracy & Loss |
| 12 | Model Evaluation |
| 13 | Save Model — SavedModel |
| 14 | Save Model — TF-Lite |
| 15 | Save Model — TFJS |
| 16 | Inference with TF-Lite |

## Results

| Metric | Value |
|---|---|
| Train Accuracy | 100.00% |
| Test Accuracy | 100.00% |
| Test Loss | 0.0002 |

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/sintiasnn/fruits360-transfer-learning.git
cd fruits360-transfer-learning
pip install -r requirements.txt
```

### 2. Run on Google Colab

Open `CNN_Fruits360_TL.ipynb` in Google Colab with GPU runtime enabled.

### 3. Kaggle API Key

Upload your `kaggle.json` when prompted, or manually place the dataset in the correct directory.

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)
