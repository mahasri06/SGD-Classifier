# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start
#### Step 2: Initialize Parameters: Start by initializing the parameters (weights) theta with random values or zeros.
#### Step 3: Compute Sigmoid Function: Define the sigmoid function that maps any real-valued number to a value between 0 and 1.
#### Step 4: Compute Loss Function: Define the loss function, which measures the error between the predicted output and the actual output.
#### Step 5: Gradient Descent Optimization: Implement the gradient descent algorithm to minimize the loss function. In each iteration, compute the gradient of the loss function with respect to the parameters (theta), and update the parameters in the opposite direction of the gradient to minimize the loss.
#### Step 6: Iterate Until Convergence: Repeat the gradient descent steps for a predefined number of iterations or until convergence criteria are met. Convergence can be determined when the change in the loss function between iterations becomes very small or when the parameters (theta) stop changing significantly.
#### Step 7: End

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Mahasri P
RegisterNumber: 212223100029
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/admin/Downloads/ml/Placement_Data.csv")
dataset

dataset = dataset.drop ('sl_no', axis=1)
dataset = dataset.drop ('salary', axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')

dataset.dtypes

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y

theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iters):
    m=len(y)
    for i in range (num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha *gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iters=1000)

def predict (theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/487464d9-5c6a-43d2-95cf-09785244bce9)

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/02d7d733-e076-4c9d-bf4a-2b5dc66cc112)

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/4b0578b0-97b6-4be9-8298-2c99e3232d04)

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/537d5750-5fdd-4b4d-92b2-3b5d7e66c02c)

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/cc068f05-db27-48a0-9690-801a6cf33e4a)

![image](https://github.com/Sajetha13/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138849316/47683fbf-7aec-44e0-ac5b-3d61fe983dd3)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

