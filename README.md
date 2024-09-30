# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DHARANISH MS
RegisterNumber: 212223240027
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:

![WhatsApp Image 2024-09-27 at 13 23 10_5bea5f2a](https://github.com/user-attachments/assets/7d63c0ef-81cb-4777-920c-6dcecc4520a9)

![WhatsApp Image 2024-09-27 at 13 23 11_54864215](https://github.com/user-attachments/assets/0485d80d-03cc-4171-a348-f76245c5e003)

![WhatsApp Image 2024-09-27 at 13 23 13_ca2bcc3a](https://github.com/user-attachments/assets/3376dd73-d0a8-4f6b-b36b-96c0691a2572)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
