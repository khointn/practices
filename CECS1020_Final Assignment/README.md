# CECS1020 Final Project - Titanic Prediction

## Introduction

This is the Group project for CECS1020 class - Introduction to Machine Learning of VinUniversity. We are group 6:

* Nguyen Tiet Nguyen Khoi
* Nguyen Duong Tung
* Nguyen Hoang Trung Dung


This zip folder includes:
* Final report (using Latex)
* ipynb file for the coding implementation
* Slides for the group presentation


Additional links:
* [Our working Google Colab folder](https://drive.google.com/drive/folders/17aytyP0m-z9awW04v71VOnRkTjKtGOgA)
* [Kaggle Profile](https://www.kaggle.com/khointn/competitions)
* [Presentation Video](https://drive.google.com/file/d/1TKtsZcjgATvDocsHnyq3eo8XjxksZAEP/view?usp=sharing)
* [Google Slides](https://docs.google.com/presentation/d/1zmoag6ffbF7Gd2EAPUQgFm2DWUOMHCbNLJk1kqL000U/edit?usp=sharing)
* [Github for zip file](https://github.com/TDung939/CECS1020)

Note: 
* Due to the report's length requirement, we could not put everything we have done into it. Please check through our ipynb file for full implementation.
* We made our slides on Google Slide. When converting it into the Powerpoint slide, it may has some visualization errors. Please access our Google Slides link above if you encounter such errors.
* The report's length requirement is 6 pages excluding the references. However, we did 7 pages (excluding the references and 2 first pages for outline). 

## The Challege (Kaggle)

"The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc)."

## Implementation (ipynb file)

### Importing Necessary Libraries

### Preprocessing Part

### Method 1: Logistic Regression
```python
from sklearn.model_selection import train_test_split
y = train['Survived']
x = train_scaled
X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42)

#Build logistic regression model

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter = 10000)
lr.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=lr.predict(X_valid)
accuracy_score(y_valid,y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_valid,y_pred)
```

### Method 2: Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report as cr

X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42)

dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict=dtc.predict(X_valid)

confusion_matrix(y_valid,y_predict)

accuracy_score(y_valid,y_predict)

print(cr(y_valid,y_predict))
```

### Method 3: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier as rc
X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42)

rfc=rc()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_valid)
accuracy_score(y_valid,rfc_y_pred)

print(cr(y_valid,rfc_y_pred))
```

## Contributing
* Nguyen Tiet Nguyen Khoi
* Nguyen Duong Tung
* Nguyen Hoang Trung Dung
<div style="display: flex">
  <img src="https://scontent.fhan2-3.fna.fbcdn.net/v/t1.6435-9/32785054_2070154336536334_2016051476874395648_n.jpg?_nc_cat=108&ccb=1-3&_nc_sid=174925&_nc_ohc=EeF8Ff5P2esAX-kI-M8&_nc_ht=scontent.fhan2-3.fna&oh=9aec1ff0c170d4ff95373b1233e3a2a5&oe=60D25AED" alt="khoi" width="50" style="border-radius: 50%"/>
  <img src="https://scontent.fhan2-4.fna.fbcdn.net/v/t1.6435-9/121823847_1223352631391577_3676600979363877791_n.jpg?_nc_cat=104&ccb=1-3&_nc_sid=09cbfe&_nc_ohc=N0wWLbZB43IAX9CsCYQ&_nc_ht=scontent.fhan2-4.fna&oh=f867d24b1489726570dc6c0155c877cd&oe=60D1E799" alt="tung" width="50" style="border-radius: 50%"/>
  <img src="https://scontent.fhan2-4.fna.fbcdn.net/v/t1.6435-9/116530643_2064712877007106_1725354370916682272_n.jpg?_nc_cat=105&ccb=1-3&_nc_sid=09cbfe&_nc_ohc=fKkR-qxbT-cAX9Kpf-J&tn=Da9lVVXxgSg0tJb9&_nc_ht=scontent.fhan2-4.fna&oh=d658c33ef444bc4954c500a63291321b&oe=60D1FD29" alt="dung" width="50" style="border-radius: 50%"/>
</div>

## License
[MIT](https://choosealicense.com/licenses/mit/)
