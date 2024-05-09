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
Developed by: 
RegisterNumber:

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
# dATA:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/e5653915-1e5d-4852-b748-dc64eaf59fc6)
# DATA INFO:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/9c999173-26e3-4dfa-8b0f-5f5cf79a25f2)
# Y PRED:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/7d10307c-160d-4865-aa68-fac5132b20b7)
# ACCURACY:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/d498b0bf-1016-4c9d-8f36-9f39968823c6)
# CONFUSION MATRIX:
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/b295cc2c-7e4a-427f-8ea4-248492f4cbe5)
# CLASSIFICATION REPORT
![image](https://github.com/shrenidhi28/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155261096/94ded670-d399-4ceb-bc83-13d0e85220ad)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
