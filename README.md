## EX 2 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dhivyapriya.R
RegisterNumber:  212222230032

```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

# Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## 1. df.head()

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/f5a22ffc-4552-4c8e-9dd5-3822fe4b3585)

## df.tail()

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/ba378a1f-1f36-428b-8597-597860388a05)

## 3. Array value of X

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/5351eadb-22d1-4cd6-bb77-692176926c70)

## 4. Array value of Y

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/ede676aa-7c78-4475-b37c-1ec95830804b)

## 5. Values of Y prediction

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/4d4ede1e-bad6-4eaa-abbc-4314066c7e8f)


## 6. Array values of Y test

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/116d8299-db97-4420-9cc1-9db2211d6905)

## 7. Training Set Graph

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/374eee9a-d29a-4037-aadb-68a64162b7c4)

## 8. TEST SET GRAPH

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/3f7b093a-6a18-46f2-ac3d-9d4a55045751)

## 9. Values of MSE, MAE and RMSE

![image](https://github.com/dhivyapriyar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477552/96c9088e-b159-4c0d-b008-71c374961e44)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
