# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### STEP 1: 
Import Necessary Libraries and Load Data
#### STEP 2:
Split Dataset into Training and Testing Sets
#### STEP 3:
Train the Model Using Stochastic Gradient Descent (SGD)
#### STEP 4:
Make Predictions and Evaluate Accuracy
#### STEP 5:
Generate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: ANU RADHA N
RegisterNumber:  212223230018
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df= pd.DataFrame(data= iris.data , columns = iris.feature_names)
df['target'] = iris.target

print(df.head())

X= df.drop('target' , axis=1)
y= df['target']

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2 , random_state=42)

sgd_clf = SGDClassifier(max_iter =1000 , tol=1e-3)

sgd_clf.fit(X_train , y_train)

y_pred =  sgd_clf.predict(X_test) 

accuracy= accuracy_score(y_test , y_pred)

print(f"Accuracy: {accuracy:.3f}")

cm= confusion_matrix(y_test, y_pred) 
print("Confusion Matrix: ") 
print(cm)
```

## Output:

![image](https://github.com/user-attachments/assets/878f9172-d8c6-4ef9-a3ac-a75ede16be7f)

![image](https://github.com/user-attachments/assets/52246347-5d66-4914-9712-3f2dbc71e095)

![image](https://github.com/user-attachments/assets/3ce012f1-dfdd-4a56-9c33-74501bf1035e)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
