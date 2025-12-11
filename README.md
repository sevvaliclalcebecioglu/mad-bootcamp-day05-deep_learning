# Deep Learning & Artificial Neural Networks Projects
**Topics Covered:** Deep Learning, Artificial Neural Networks, CNN, RNN, LSTM, GAN, Computer Vision, TensorFlow, PyTorch

---

## Table of Contents
1. [Introduction](#introduction)
2. [Deep Learning Concepts](#deep-learning-concepts)
3. [Projects](#projects)
   - [Classification Project: Pima Indians Diabetes](#classification-project-pima-indians-diabetes)
   - [Regression Project: KC House Prices](#regression-project-kc-house-prices)
   - [Early Stopping Project: Cars](#early-stopping-project-cars)
   - [Computer Vision: MNIST Digit Classification](#computer-vision-mnist-digit-classification)
4. [Tools & Libraries](#tools--libraries)
5. [License](#license)

---

## Introduction
This repository contains various deep learning projects demonstrating Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), LSTM, and GANs using Python frameworks such as TensorFlow and PyTorch. It also covers Computer Vision applications.

---

## Deep Learning Concepts
- **Deep Learning:** Multi-layer neural networks that automatically extract features from data.
- **CNN:** Used for image/video processing, extracts spatial features automatically.
- **RNN & LSTM:** Process sequential data and maintain temporal dependencies.
- **GAN:** Generator and Discriminator networks for realistic data generation.
- **Activation Functions:** ReLU, Leaky ReLU, Sigmoid, Tanh.
- **Loss Functions:** Binary Cross-Entropy (classification), Mean Squared Error (regression).
- **Hyperparameters:** Learning rate, batch size, epochs, layer sizes.
- **Training:** Backpropagation and gradient descent for weight optimization.

---

## Projects

### Classification Project: Pima Indians Diabetes
- **Dataset:** `pima-indians-diabetes.csv`
- **Goal:** Predict diabetes (yes/no) based on health measurements.
- **Libraries:** `pandas`, `tensorflow.keras`, `sklearn`
- **Model:** Sequential ANN with multiple Dense layers.
- **Training:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(80, activation='relu'),
    Dense(120, activation='relu'),
    Dense(80, activation='relu'),
    Dense(30, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, batch_size=32, validation_split=0.1, epochs=100)
