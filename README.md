# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries and Load the Dataset  
2. Preprocess the Data (Remove Unwanted Columns and Handle Missing/Duplicate Values)  
3. Encode Categorical Variables using LabelEncoder  
4. Split the Data into Training and Testing Sets and Train the Logistic Regression Model  
5. Evaluate the Model using Accuracy, Confusion Matrix, and Classification Report  

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GAYATHRI.C
RegisterNumber: 250009125

import pandas as pd 
data=pd.read_csv("Placement_Data.csv") 
print("first 5 elements:\n",data.head()) 
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1) 
data1.head() 
data1.isnull() 
data1.duplicated().sum() 
from sklearn .preprocessing import LabelEncoder 
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"]) 
data1 
x=data1.iloc[:,:-1] 
print("\n x:\n",x) 
y=data1["status"] 
print("\n y:\n",y)
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train) 
y_pred=lr.predict(x_test) 
print("\n y_pred:\n",y_pred) 
from sklearn.metrics import accuracy_score 
accuracy=accuracy_score(y_test,y_pred) 
print("\n accuracy:\n",accuracy) 
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\n confusion matrix:\n",confusion)
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("\n classification_report1:\n",classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 

```

## Output:
<img width="817" height="339" alt="Screenshot 2025-10-06 141327" src="https://github.com/user-attachments/assets/d01ecda9-4fbb-4e82-bca5-46039baef9b7" />
<img width="1099" height="636" alt="Screenshot 2025-10-06 141337" src="https://github.com/user-attachments/assets/485a1ddb-b0d1-476e-802a-9ab1b5effcbb" />
<img width="660" height="304" alt="Screenshot 2025-10-06 141344" src="https://github.com/user-attachments/assets/4ce86343-be94-4d39-982b-cae38bec626c" />
<img width="831" height="88" alt="Screenshot 2025-10-06 141349" src="https://github.com/user-attachments/assets/ede26409-4ec2-4761-9e3e-dbf4eec64cdc" />
<img width="570" height="376" alt="Screenshot 2025-10-06 141409" src="https://github.com/user-attachments/assets/496f0249-5958-4973-8a2e-1866774889bd" />

## PREDICTION:
<img width="173" height="24" alt="Screenshot 2025-10-06 141423" src="https://github.com/user-attachments/assets/cd74f4ea-f4ea-476b-a352-78f4ab6b4716" />







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
