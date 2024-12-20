# Regularized Logistic Regression for Microchip Quality Assurance

## Overview
This project implements a regularized logistic regression model to predict whether microchips pass quality assurance (QA) based on their test results. The solution includes visualizing the dataset, mapping features to higher dimensions, and applying regularization to combat overfitting.

---

## Problem Statement

You are tasked with determining whether microchips should be accepted or rejected based on two QA test results. A dataset with test results and corresponding acceptance decisions is provided to train a logistic regression model.

---

## Data Visualization

The dataset contains two test scores (`X_train`) and their corresponding outcomes (`y_train`). The microchips are classified as:
- `y_train = 1` (Accepted)
- `y_train = 0` (Rejected)

### Dataset Characteristics:
- `X_train` shape: (118, 2)
- `y_train` shape: (118,)

### Scatter Plot:
The following plot shows the dataset, where:
- Accepted microchips are represented with a "+" marker.
- Rejected microchips are represented with a "o" marker.

![Dataset Visualization](images/figure%203.png)

Figure 3 shows that a linear decision boundary is insufficient for this dataset.

---

## Feature Mapping

To create a non-linear decision boundary, we map features to polynomial terms using the function:

$$\text{map\_feature}(x) = \begin{bmatrix}
    x_1 \\
    x_2 \\
    x_1^2 \\
    x_1 x_2 \\
    x_2^2 \\
    \vdots \\
    x_1 x_2^5 \\
    x_2^6
\end{bmatrix}$$

This transformation increases the feature vector's dimensionality from 2 to 27.

### Example:
- Original shape: (118, 2)
- Transformed shape: (118, 27)

---

## Cost Function for Regularized Logistic Regression

The regularized logistic regression cost function is defined as:

$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left(1 - y^{(i)}\right) \log\left(1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right)\right) \right] + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

The additional regularization term:

$$\frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

helps prevent overfitting by penalizing large weights. Note that the bias term $b$ is not regularized.

### Regularized Cost Example:
- Regularized cost: **0.661825**

---

## Gradient for Regularized Logistic Regression

The gradients for the cost function are calculated as:

- For bias term $b$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})$$

- For weights $\mathbf{w}$:
  $$\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} \left[(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}\right] + \frac{\lambda}{m} w_j$$

---

## Implementation

### Key Functions:
1. **Feature Mapping**: `map_feature` transforms the input features into higher-dimensional polynomial terms.
2. **Regularized Cost Function**: `compute_cost_reg` computes the total cost with regularization.
3. **Gradient Descent with Regularization**: Updates parameters to minimize the cost function.

---

## Results

The regularized logistic regression model effectively separates the microchips into accepted and rejected categories by learning a non-linear decision boundary.

---

## Acknowledgments
This project is based on the Coursera Machine Learning course by Andrew Ng. Helper functions and datasets are included as part of the course material.

---

## References
- [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)

---

### Note
For additional details, refer to the `utils.py` file for helper functions and the dataset `ex2data2.txt`.
