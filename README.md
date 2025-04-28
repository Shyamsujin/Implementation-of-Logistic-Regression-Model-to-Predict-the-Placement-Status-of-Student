# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by  : Shyam Sujin U
RegisterNumber:  212223040201
```
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv(r'/Users/home/Desktop/Workspace/College/Semester_4/Machine_Learning/Experiments/EX05/Placement Data.csv')
df

df.info()

df.describe()

df.shape

df.isnull().sum()

df.drop(columns=['sl_no','salary'],inplace=True)

df.head()

df.shape

df.info()

df['gender']=df['gender'].astype('category')
df['ssc_b']=df['ssc_b'].astype('category')
df['hsc_b']=df['hsc_b'].astype('category')
df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df

df.info()

for i in df.columns:
    if i in ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']:
        df[i]=df[i].cat.codes

df.info()

X=df.drop(columns='status')
Y=df[['status']]

print(X)
print(Y)
print(type(Y))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

X.shape

Y.shape

X_train.shape

X_test.shape

model=LogisticRegression()

Scaler=StandardScaler()

scaled_x_train=Scaler.fit_transform(X_train)
scaled_x_test=Scaler.fit_transform(X_test)

model.fit(scaled_x_train,Y_train)

accuracy=model.predict(scaled_x_test)
score=accuracy_score(accuracy,Y_test)
print(score)

conf_mat=confusion_matrix(accuracy,Y_test)
print(conf_mat)


df.head()

input_data=(0,87,0,95,0,2,91,1,1,92,1,57)
input_as_npy_array=np.asanyarray(input_data)
input_reshaped=input_as_npy_array.reshape(1,-1)
prediction=model.predict(input_reshaped)
print(prediction)

```

## Output:
<img width="1625" alt="EXP05" src="https://github.com/user-attachments/assets/bdab0bf0-d55c-43d4-95c5-32d1dd5c2483" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
