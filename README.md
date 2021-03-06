# Project - Iris Data Set Classification (Using Django)

## Contents
* [Background](#background)
* [Data Set](#data-set)
* [Application](#application)

## Background
[Django](https://www.djangoproject.com/) is a high-level Python Web framework that helps to develop websites faster and easier. When users build a website, they always need to set similar components. Motivated by this, experienced developers teamed up and created frameworks which can alleviate the problems and make users to focus on building a new website.

## Data Set
Iris data set is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) that has 150 instances with 4 attributes, balanced class distribution and no missing values. The attributes are sepal length, sepal width, petal length and petal width. This data set contains 3 classes of 50 instances each having Iris Setosa, Iris Versicolour and Iris Virginica.

## Application
According to Machine Learning methods, we can build different classifiers toward data set. Then, we create a website (Figure 1) that user can input attributes and select classifiers based on their needs to predict iris class. This new data will be import into data base and the admin can check this classification is correct or not. If it is not, the admin can delete the result.

<img src="/image/django-iris.JPG" width="800"/> 

<em>Figure 1: Django.</em>

Users can select SVM, Decision Tree and KNN classifiers. If the inputs are wrong, the prediction cannot be showd in Figure 2.

<img src="/image/django-iris_input_error_result.JPG" width="400"/> 

<em>Figure 2: The result for error inputs.</em>

Once users get the result, the input and prediction will be listed into database. User can click "DB" (on top right) to see all data (as Figure 3).

<img src="/image/django-iris_db.JPG" width="800"/> 

<em>Figure 3: Data base.</em>
