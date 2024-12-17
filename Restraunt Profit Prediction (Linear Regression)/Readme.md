# Linear Regression (Single Variable)

---

## Outline
1. [Problem Statement](#1---problem-statement)
2. [Dataset](#2---dataset)
3. [Mathematical Concepts](#3---mathematical-concepts)
4. [Results](#4---results)
5. [Summary](#5---summary)
6. [Conclusion](#6---conclusion)


## 1 - Problem Statement

Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. 

- We would like to expand our business to cities that may give our restaurant higher profits.
- The chain already has restaurants in various cities, and we have data for profits and populations from those cities.
- We also have data on cities that are candidates for a new restaurant. 
- For these cities, we have the city population.

Can we use the data to help us identify which cities may potentially give our business higher profits?

## 2 - Dataset
1. **Dataset Overview**:
    - The dataset consists of a single feature (population) and a target variable (restaurant profit).
    - Example data: 
        | Population | Profit  |
        |------------|---------|
        | 6.12       | 17.59   |
        | 5.18       | 15.14   |
        | ...        | ...     |

2. **Data Description**

We will start by loading the dataset for this task. 
- The `load_data()` function loads the data into variables `x_train` and `y_train`.
  - `x_train`: Population of a city
  - `y_train`: Profit of a restaurant in that city. A negative value for profit indicates a loss.
  - Both `x_train` and `y_train` are numpy arrays.

## 3 - Mathematical Concepts

In this project, we aim to fit the parameters $w$ (slope) and $b$ (intercept) of a linear regression model to our dataset, which involves predicting the monthly profit of a restaurant based on the population of a city.

The linear regression model can be expressed as:

$$
f_{w,b}(x) = w \cdot x + b
$$

Where:
- $x$ is the input feature (city population),
- $f_{w,b}(x)$ is the predicted monthly profit,
- $w$ is the slope (parameter we need to find),
- $b$ is the intercept.

### Training the Model

To train the model, we need to determine the values of $w$ and $b$ that minimize the difference between the model's predictions and the actual profits in the dataset. This is achieved by using a **cost function** $J(w,b)$, which evaluates how well the model's predictions match the actual data. The goal is to find the values of $w$ and $b$ that minimize the cost function.

The cost function for linear regression is defined as:

$$
J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

Where:
- $f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$ is the model's prediction for the $i^{th}$ city’s profit,
- $y^{(i)}$ is the actual profit recorded for the $i^{th}$ city,
- $m$ is the number of training examples.

The cost function calculates the squared difference between the predicted profit and the actual profit for each training example. The sum of these squared differences is then averaged, and the factor $\frac{1}{2m}$ is included for easier differentiation during optimization.

### Gradient Descent

Once the cost function is defined, we need to minimize it to find the optimal values of $w$ and $b$. This is where **gradient descent** comes in, an iterative optimization algorithm that adjusts the parameters $w$ and $b$ in small steps to reduce the cost.

In each step of gradient descent, we update the parameters using the following formulas:

$$
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

$$
w := w - \alpha \frac{\partial J(w,b)}{\partial w}
$$

Where:
- $\alpha$ is the learning rate (step size),
- $\frac{\partial J(w,b)}{\partial b}$ and $\frac{\partial J(w,b)}{\partial w}$ are the partial derivatives of the cost function with respect to $b$ and $w$, respectively.

These updates move the parameters $w$ and $b$ in the direction that reduces the cost function.

#### Computing the Gradients

To update the parameters, we need to compute the partial derivatives of the cost function:

$$
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

$$
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

These partial derivatives represent the gradients of the cost function with respect to $b$ and $w$. The gradient descent algorithm uses these gradients to update the parameters in the direction that reduces the cost function.

## 4 - Results
The model predicts restaurant profit based on population. The final results include the optimized parameters $( \theta_0 )$ and $( \theta_1 )$, along with a graph showing the fit of the linear regression model to the data.

![image](https://github.com/user-attachments/assets/10854f67-89a3-459a-8c27-6a5ca66eae0b)
![image](https://github.com/user-attachments/assets/aaa9b2f0-bd74-42d1-a69c-6aeee8b480f4)


## 5 - Summary

In summary, this linear regression model predicts the monthly profit for a restaurant based on the population of a city. The goal is to find the values of $w$ (slope) and $b$ (intercept) that minimize the cost function $J(w,b)$. This is achieved by iteratively adjusting the parameters using **gradient descent**, which updates $w$ and $b$ based on the gradients computed from the cost function.

## 6 - Conclusion

This lab demonstrates the application of linear regression and gradient descent to predict restaurant profits based on city population. By fitting the model to the data and minimizing the cost function, We can determine the best parameters for our  model, helping We make data-driven decisions on where to open new restaurant outlets.





