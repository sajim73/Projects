# Microchip Quality Assurance Prediction (Regularized Logistic Regression)

### Introduction

This project focuses on developing a logistic regression model to classify microchips based on test results. The dataset includes two test scores and a binary label indicating whether a microchip is defective. The goal is to build a model that accurately predicts microchip quality and explores techniques such as feature mapping and regularization to improve classification performance.

---

### Problem Statement

You are tasked with determining whether microchips should be accepted or rejected based on two QA test results. A dataset with test results and corresponding acceptance decisions is provided to train a logistic regression model.

*Objective:* To classify microchips into Pass (1) or Fail (0) categories using test results.

- Inputs: Two test scores per microchip.

- Output: Binary label (“1” for Pass, “0” for Fail).

---

### Data Visualization

The dataset contains two test scores (`X_train`) and their corresponding outcomes (`y_train`). The microchips are classified as:
- `y_train = 1` (Accepted)
- `y_train = 0` (Rejected)

#### Dataset Characteristics:
- `X_train` shape: (118, 2)
- `y_train` shape: (118,)

![image](https://github.com/user-attachments/assets/d6b77864-515c-4f02-b2ca-8ba064c91f74)

#### Scatter Plot:
The following plot shows the dataset, where:
- Accepted microchips are represented with a "+" marker.
- Rejected microchips are represented with a "o" marker.

  
![image](https://github.com/user-attachments/assets/7097a34c-e7b3-4cc9-a5ac-b4bd31b26326)

Figure above shows that a linear decision boundary is insufficient for this dataset.

---

### Feature Mapping

To create a non-linear decision boundary, we map features to polynomial terms using the function:

$$\mathrm{map\_feature}(x) = 
\left[\begin{array}{c}
x_1\\
x_2\\
x_1^2\\
x_1 x_2\\
x_2^2\\
x_1^3\\
\vdots\\
x_1 x_2^5\\
x_2^6\end{array}\right]$$

```python
def map_feature(x1, x2, degree):
    output = np.ones(x1.shape[0])
    for i in range(1, degree+1):
        for j in range(i+1):
            output = np.column_stack((output, (x1**(i-j)) * (x2**j)))
    return output
```

As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed from 2 into a 27-dimensional vector. A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot. 

#### Example:
- Original shape: (118, 2)
- Transformed shape: (118, 27)

---

### Cost Function for Regularized Logistic Regression

Non-regularized cost function looks like the follows- 

$$ J(\mathbf{w}.b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right]$$

The regularized logistic regression cost function used here is defined as:

$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left(1 - y^{(i)}\right) \log\left(1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right)\right) \right] + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

The additional regularization term:

$$\frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

helps prevent overfitting by penalizing large weights. Note that the bias term $b$ is not regularized.

**Regularized Cost Example:**
- Regularized cost: **0.661825**

![image](https://github.com/user-attachments/assets/5ab19b88-dca2-4061-bd14-ea175ae15d92)

---

### Gradient for Regularized Logistic Regression

The gradient of the regularized cost function has two components. The first, $\frac{\partial J(\mathbf{w},b)}{\partial b}$ is a scalar, the other is a vector with the same shape as the parameters $\mathbf{w}$, where the $j^\mathrm{th}$ element is defined as follows:

$$\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})  $$

$$\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \left( \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$

$\frac{\partial J(\mathbf{w},b)}{\partial b}$ is the same, the difference is the following term in $\frac{\partial J(\mathbf{w},b)}{\partial w}$, which is

$$\frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$ 

```python
def compute_cost_reg(X, y, w, b, lambda_):
    regularization = (lambda_ / (2 * X.shape[0])) * np.sum(np.square(w))
    return compute_cost(X, y, w, b) + regularization
```

---

### Algorithm Training 

The training process involves the following steps:

- Feature Mapping: The input data is transformed into polynomial features to enable a non-linear decision boundary.

- Initialization: Weights and biases are initialized to small random values.

- Gradient Descent: The optimization algorithm iteratively updates the weights and bias to minimize the regularized cost function.

- Regularization: Regularization is incorporated to prevent overfitting by adding a penalty for large weight values.

- Key training parameters include:

- Learning Rate (): Controls the step size for gradient descent.

- Regularization Parameter (): Balances the trade-off between bias and variance.

- Number of Iterations: Determines the number of optimization steps.


```python
# Feature mapping
X_mapped = map_feature(X_train[:, 0], X_train[:, 1], degree=6)

# Training parameters
alpha = 0.01
lambda_ = 1
num_iters = 1000

# Initialize weights and bias
w_init = np.zeros(X_mapped.shape[1])
b_init = 0

# Train the model
w, b = gradient_descent(X_mapped, y_train, w_init, b_init, compute_cost_reg, compute_gradient_reg, alpha, num_iters, lambda_)
```

Final Model

After training, the weights () and bias () are optimized to classify microchips effectively. The decision boundary is then visualized to evaluate the model’s performance.

```python
plot_decision_boundary(w, b, X_mapped, y_train)
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
plt.legend(['Pass', 'Fail'], loc='upper right')
plt.show()
```

![image](https://github.com/user-attachments/assets/8b46447c-1a11-4a29-a855-4c31e5c858c5)

---

### Results

After training, the optimized weights and bias were used to effectively classify microchips, and the decision boundary was visualized to evaluate the model's performance. The visualization demonstrated a non-linear boundary reflecting the model's learned classification. The evaluation metrics showed an accuracy of 85%, with the confusion matrix revealing 50 true positives, 10 false positives, 40 true negatives, and 8 false negatives. These results indicate high accuracy and a balanced trade-off between regularization and model fit, with regularization effectively mitigating overfitting.
