# Neural Networks for Handwritten Digit Recognition

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Dependencies](#setup-and-dependencies)
3. [ReLU Activation](#relu-activation)
4. [Softmax Function](#softmax-function)
5. [Neural Network Architecture](#neural-network-architecture)
   - [Binary Classification](#binary-classification)
   - [Multiclass Classification](#multiclass-classification)
6. [Dataset Details](#dataset-details)
   - [Binary Classification](#binary-classification-1)
   - [Multiclass Classification](#multiclass-classification-1)
7. [Implementation Details](#implementation-details)
   - [Training Parameters](#training-parameters)
8. [Evaluation](#evaluation)
   - [Binary Classification](#binary-classification-2)
   - [Multiclass Classification](#multiclass-classification-2)
9. [Results](#results)
   - [Binary Classification](#binary-classification-3)
   - [Multiclass Classification](#multiclass-classification-3)
10. [Usage](#usage)
11. [References](#references)

---

## Project Overview
This project demonstrates the use of neural networks for recognizing handwritten digits. It includes both binary classification between digits `0` and `1` and multiclass classification for digits `0-9`. Handwritten digit recognition is a widely-used application of machine learning, from reading postal codes to processing bank checks. This project utilizes TensorFlow and Keras to build, train, and evaluate neural network models for these tasks.

---

## Setup and Dependencies
Before running the code, ensure you have the following dependencies installed:

```bash
pip install tensorflow matplotlib numpy
```

Additionally, import the necessary libraries in your script:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
```

---

## ReLU Activation
The ReLU (Rectified Linear Unit) activation function is used in hidden layers of the multiclass classification model to introduce non-linearity and prevent issues like vanishing gradients.

**ReLU Function:**
$f(x) = \max(0, x)$

---

## Softmax Function
For multiclass classification, the softmax activation function is applied in the output layer to convert logits into probabilities for each class.

**Softmax Function:**
$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}$

---

## Neural Network Architecture

### Binary Classification
| Layer | Units | Activation |
|-------|-------|------------|
| Input | 400   | -          |
| Dense | 25    | Sigmoid    |
| Dense | 15    | Sigmoid    |
| Dense | 1     | Sigmoid    |

**Code:**
```python
model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(25, activation='sigmoid'),
    Dense(15, activation='sigmoid'),
    Dense(1, activation='sigmoid')
], name="binary_nn_model")
model.summary()
```

### Multiclass Classification
| Layer | Units | Activation |
|-------|-------|------------|
| Input | 784   | -          |
| Dense | 128   | ReLU       |
| Dense | 64    | ReLU       |
| Dense | 10    | Softmax    |

**Code:**
```python
model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
], name="multiclass_nn_model")
model.summary()
```
![image](https://github.com/user-attachments/assets/a4c34b30-d97a-432f-865c-c3b7a840c1fb)

---

## Dataset Details

### Binary Classification
- **Input:** A 1000x400 matrix `X`, where each row represents an image as a flattened 400-dimensional vector.
- **Labels:** A 1000x1 vector `y`, where `y=0` represents digit `0` and `y=1` represents digit `1`.

![image](https://github.com/user-attachments/assets/98f97ba5-8389-4e6b-802d-1bfe17246ad7)

### Multiclass Classification
- **Input:** A 70,000x784 matrix `X`, where each row represents an image as a flattened 784-dimensional vector.
- **Labels:** A 70,000x1 vector `y`, where `y` ranges from `0` to `9`.

![image](https://github.com/user-attachments/assets/d983e9a3-da09-4c7b-8bbd-8e43f3c1bf2e)


```python
# Example visualization
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(X.shape[0])
    ax.imshow(X[random_index].reshape((28, 28)), cmap='gray')  # Adjust for dataset type
    ax.set_title(f"Label: {y[random_index]}")
    ax.axis('off')
plt.show()
```

---

## Implementation Details

### Training Parameters
#### Binary Classification
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (Learning Rate = 0.001)
- **Epochs:** 20

![image](https://github.com/user-attachments/assets/b31606b3-69c5-451d-a116-ec9d2e570b09)

#### Multiclass Classification
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (Learning Rate = 0.001)
- **Epochs:** 20

**Code Example:**
```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy() if binary else tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
```

Vectorized NumPy Model Implementation
The optional lectures described vector and matrix operations that can be used to speed the calculations. Below describes a layer operation that computes the output for all units in a layer on a given input example:

![image](https://github.com/user-attachments/assets/f3dc7ae0-4729-46f1-856f-b5c851b7261d)

We can demonstrate this using the examples X and the W1,b1 parameters above. We use np.matmul to perform the matrix multiply. Note, the dimensions of x and W must be compatible as shown in the diagram above.

```python
x = X[0].reshape(-1,1)         # column vector (400,1)
z1 = np.matmul(x.T,W1) + b1    # (1,400)(400,25) = (1,25)
a1 = sigmoid(z1)
print(a1.shape)
(1, 25)
```

You can take this a step further and compute all the units for all examples in one Matrix-Matrix operation.

![image](https://github.com/user-attachments/assets/e1214a5b-459d-4552-9fae-1ee8b0ac757e)


---

## Evaluation

### Binary Classification
- **Metrics:** Loss and Accuracy
- **Thresholding:** Probabilities above `0.5` are classified as `1`, below as `0`.

### Multiclass Classification
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Class Predictions:** Use `argmax` to find the class with the highest probability.

**Suggested Visuals:** Include confusion matrices and classification reports.
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Example for confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

---

## Results

### Binary Classification
- **Accuracy:** Achieved ~98% on test data.
- **Insights:** Correctly distinguishes digits `0` and `1` with minimal false positives/negatives.

### Multiclass Classification
- **Accuracy:** Achieved ~95% on test data.
- **Insights:** Common misclassifications occur for visually similar digits like `8` and `3`.

---

## Usage

1. Clone the repository.
2. Install dependencies using the provided command.
3. Run the scripts to train and evaluate models.

---

## References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)




---


