# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: C.SHRENIDHI
RegisterNumber: 212223040196
```

```
import pandas as pd
data= pd.read_csv("spam.csv",encoding="Windows-1252")
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train= cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred= svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```





## OUTPUT:
![SVM For Spam Mail Detection](sam.png)
# DATA:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/d02bf714-d32e-4724-b490-89fc5a731d82)

# DATA INFO:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/a9bb9790-d565-4b4b-b80b-a2ec97afa7b4)

# Y PRED:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/ae874e14-60a6-4f76-8227-3fd3d536888b)

# ACCURACY:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/48b2e57e-639a-40e9-91d2-885d4b30b2da)

# CONFUSION MATRIX:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/0726a229-081f-4ed5-aa00-ba31464150e5)

# CLASSIFICATION REPORT
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/cf0c6d3c-b89a-4035-b249-f8d874a79ba2)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
