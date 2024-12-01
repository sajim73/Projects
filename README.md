# Projects


# Linear Regression (One Variable)

## Outline
1. [Problem Statement](#1---problem-statement)
2. [Dataset](#2---dataset)
3. [Refresher on Linear Regression](#3---refresher-on-linear-regression)
4. [Compute Cost](#4---compute-cost)
5. [Gradient Descent](#5---gradient-descent)
6. [Learning Parameters Using Batch Gradient Descent](#6---learning-parameters-using-batch-gradient-descent)

---

## 1 - Problem Statement

Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. 

- You would like to expand your business to cities that may give your restaurant higher profits.
- The chain already has restaurants in various cities, and you have data for profits and populations from those cities.
- You also have data on cities that are candidates for a new restaurant. 
    - For these cities, you have the city population.

Can you use the data to help you identify which cities may potentially give your business higher profits?

---

## 2 - Dataset

We will start by loading the dataset for this task. 
- The `load_data()` function loads the data into variables `x_train` and `y_train`.
  - `x_train`: Population of a city
  - `y_train`: Profit of a restaurant in that city. A negative value for profit indicates a loss.
  - Both `x_train` and `y_train` are numpy arrays.

---

## 3 - Refresher on Linear Regression

In this practice lab, you will fit the linear regression parameters \(w\) and \(b\) to your dataset.

The model function for linear regression, which maps from \(x\) (city population) to \(y\) (monthly profit for that city), is represented as:

\[
f_{w,b}(x) = w \cdot x + b
\]

To train a linear regression model, you need to find the best parameters \(w\) (slope) and \(b\) (intercept) that fit your dataset.

To compare how one choice of \(w\) and \(b\) is better or worse than another, you can evaluate it using a cost function \(J(w,b)\).

The choice of \(w\) and \(b\) that fits your data the best is the one that has the smallest cost \(J(w,b)\).

To find the values of \(w\) and \(b\) that minimize the cost function \(J(w,b)\), you can use a method called gradient descent. With each step of gradient descent, your parameters \(w\) and \(b\) get closer to the optimal values that will achieve the lowest cost \(J(w,b)\).

The trained linear regression model can then take the input feature \(x\) (city population) and output a prediction \(f_{w,b}(x)\) (predicted monthly profit for a restaurant in that city).

---

## 4 - Compute Cost

Gradient descent involves repeated steps to adjust the value of your parameters \(w\) and \(b\) to gradually minimize the cost \(J(w,b)\).

At each step of gradient descent, it is helpful to compute the cost \(J(w,b)\) as \(w\) and \(b\) are updated to monitor progress.

### Cost Function

For one variable, the cost function for linear regression is defined as:

\[
J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
\]

Where:
- \( f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b \) is the model's prediction for the \(i^{th}\) city’s profit.
- \( y^{(i)} \) is the actual profit recorded in the data for the \(i^{th}\) city.
- \( m \) is the number of training examples in the dataset.

### Model Prediction

For linear regression with one variable, the prediction of the model \( f_{w,b} \) for an example \( x^{(i)} \) is represented as:

\[
f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b
\]

This is the equation of a line, where \(b\) is the intercept and \(w\) is the slope.

### Implementation

Complete the `compute_cost()` function to compute the cost \(J(w,b)\).

---

## 5 - Gradient Descent

Gradient descent is used to minimize the cost function \(J(w,b)\) by adjusting the parameters \(w\) and \(b\) iteratively. The update rule for gradient descent is:

\[
\text{repeat until convergence:}
\]
\[
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
\]
\[
w := w - \alpha \frac{\partial J(w,b)}{\partial w}
\]

Where:
- \( \alpha \) is the learning rate (step size).
- \( \frac{\partial J(w,b)}{\partial b} \) and \( \frac{\partial J(w,b)}{\partial w} \) are the partial derivatives of the cost function with respect to \(b\) and \(w\), respectively.

---

## 6 - Learning Parameters Using Batch Gradient Descent

In this section, you will implement the gradient descent algorithm for learning the parameters \(w\) and \(b\) using batch gradient descent.

The partial derivatives for the cost function are:

\[
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
\]

\[
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
\]

Where \(m\) is the number of training examples. You will implement a function called `compute_gradient()` that calculates these partial derivatives.


### Next Steps:
1. Implement the `compute_cost()` function to calculate the cost \( J(w,b) \).
2. Implement the `compute_gradient()` function to calculate the partial derivatives of the cost function.
3. Implement the gradient descent algorithm to update the parameters \( w \) and \( b \) and minimize the cost function.

---

## Conclusion

This lab demonstrates the application of linear regression and gradient descent to predict restaurant profits based on city population. By fitting the model to the data and minimizing the cost function, you can determine the best parameters for your model, helping you make data-driven decisions on where to open new restaurant outlets.
