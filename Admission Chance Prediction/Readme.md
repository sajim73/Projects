# Admission Decision Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Visualization of Data](#visualization-of-data)
4. [Mathematical Foundation](#mathematical-foundation)
    - [Sigmoid Function](#sigmoid-function)
    - [Cost Function](#cost-function)
    - [Gradient Descent](#gradient-descent)
5. [Algorithm Training](#algorithm-training)
6. [Results](#results)
---
## Introduction
This project implements a logistic regression model to predict whether a student will be admitted to a university based on their scores in two exams. The following sections outline the key components, including the problem statement, the sigmoid function, the cost function, and the gradient computation. All these elements are central to the process of training the model.

---

## Problem Statement
You are tasked with estimating the probability of university admission for applicants based on their exam scores. Historical data from previous applicants serves as the training set, where:
- **Input:** Two exam scores.
- **Output:** Admission decision (1 = Admitted, 0 = Not Admitted).

Your goal is to build a classification model to predict these outcomes.

---

## Visualization of Data
- **Data Description:**
  - `X_train`: Exam scores (features).
  - `y_train`: Admission decisions (labels).

```python
X_train, y_train = load_data("data/ex2data1.txt")
```

- **Visualization:**
  Data points are plotted on a 2D graph with different markers for admitted and not-admitted candidates.

![image](https://github.com/user-attachments/assets/25affdc1-26ee-481c-ad21-eed5578c48a2)

```python
plot_data(X_train, y_train, pos_label="Admitted", neg_label="Not admitted")
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()
```

---
## Mathematical Foundation

The logistic regression model is built upon three main components: the sigmoid function, the cost function, and gradient descent.

### Sigmoid Function

The logistic regression model is based on the sigmoid function, which outputs a probability between 0 and 1. The sigmoid function is defined as:

$$
    g(z) = \frac{1}{1 + e^{-z}}
$$

The model prediction is calculated as:

$$
    f_{w,b}(x) = g(w \cdot x + b)
$$

The sigmoid function is implemented as follows:

```python
import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
```

The sigmoid function is applied element-wise when `z` is an array, producing the predicted probabilities for all training examples.

To test the sigmoid function:

```python
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid([-1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))
```

Expected Output:
```
sigmoid(0) = 0.5
sigmoid([-1, 0, 1, 2]) ≈ [0.269, 0.5, 0.731, 0.881]
```

---

### Cost Function

The cost function for logistic regression measures how well the model's predictions match the actual labels. The cost function is defined as:

$$
    J(w,b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log(f_{w,b}(x^{(i)})) - (1-y^{(i)}) \log(1-f_{w,b}(x^{(i)})) \right]
$$

Where:
- $( m )$: Number of training examples.
- $( f_{w,b}(x^{(i)}) )$: Model prediction for the $( i )$-th example, calculated using the sigmoid function.

The cost function is implemented as:

```python
def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    total_cost = cost / m
    return total_cost
```

To test the cost function:

```python
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.0
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))

test_w = np.array([0.2, 0.2])
test_b = -24.0
cost = compute_cost(X_train, y_train, test_w, test_b)
print('Cost at test w,b: {:.3f}'.format(cost))
```
![image](https://github.com/user-attachments/assets/76774a84-4bcf-4e02-8f05-ded195efece8)

---

## Gradient Descent
Gradient descent updates the model parameters to minimize the cost function:

$$\begin{align*}
& b := b - \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
& w_j := w_j - \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j}, \; \text{for } j = 0 \text{ to } n-1
\end{align*}$$

The gradient of the cost function with respect to the model parameters $( w )$ and $( b )$ is computed to guide the optimization process. The gradients are defined as:

##### Gradient with respect to \( b \):
$$
    \frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

##### Gradient with respect to \( w_j \):
$$
    \frac{\partial J(w,b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$


```python
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dw = np.zeros(n)
    db = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        dw += (f_wb - y[i]) * X[i]
        db += (f_wb - y[i])

    dw /= m
    db /= m
    return dw, db
```

![image](https://github.com/user-attachments/assets/1e5ff131-b204-4f6e-b738-1f3e424b1bb7)


---

## Algorithm Training
Using the gradient descent algorithm, train the model by iteratively updating $\mathbf{w}$ and $b$.

- **Code Implementation:**
```python
def gradient_descent(X, y, w, b, cost_function, gradient_function, alpha, num_iters):
    for _ in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b
```
---
## Results

With the sigmoid function, cost function, and gradients implemented, the model was optimized using gradient descent to minimize the cost function. The model was trained to predict university admission based on exam scores. After several iterations of gradient descent, the final model achieved an accuracy of 92% on the test set, demonstrating its effectiveness in predicting the likelihood of admission based on the given features.
