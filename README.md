## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### 1.Import the standard Libraries. 
### 2.Set variables for assigning dataset values. 
### 3.Import linear regression from sklearn. 
### 4.Assign the points for representing in the graph. 
### 5.Predict the regression for marks by using the representation of the graph.
### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: STANLEY S 
RegisterNumber: 212223110054
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## Dataset:
![image](https://github.com/user-attachments/assets/126b985c-cb5a-466a-a1f5-8d60d7c7a404)
## Head values:
![image](https://github.com/user-attachments/assets/92d7a440-aa9a-4eae-8f38-95c08b3a5d0f)
## Tail values:
![image](https://github.com/user-attachments/assets/c133885b-0e40-4841-aab8-0cdc5b84e2e6)
## x and y values:
![image](https://github.com/user-attachments/assets/69638f54-76df-4eb4-a322-00ebb67ad1e0)
## Prediction values of x and y:
![image](https://github.com/user-attachments/assets/83a803d8-5022-4d1c-887b-341a1cbd0396)
## MSE,MAE,RMSE
![image](https://github.com/user-attachments/assets/d025f0ea-d7c6-47ab-808a-920ce06e38c4)
## Training set:
![image](https://github.com/user-attachments/assets/26c07da0-3a35-469f-b9fe-103eaf8f3f4f)

## Testing set:
![image](https://github.com/user-attachments/assets/e8af38a8-8565-42d9-9de1-bd2d4409160d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
