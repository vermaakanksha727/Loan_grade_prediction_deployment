# Logistic Regession to Predict Loan Grade
**Logistic Regression** is a Machine Learning classification algorithnm that is used to predict the probability of categorical dependent variable.In logistic regression,the dependent variable is a **binary** or **Multinomial**, which involves having more than one category.

## Logistic Regression Assumptions:
- The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
- The independent variables are linearly related to the log odds.
- Logistic regression requires quite large sample sizes.

> **🧮 Mathematics behind logistic regression**
>
> Remember how linear regression often used ordinary least squares to arrive at a value? Logistic regression relies on the concept of 'maximum likelihood' using [sigmoid functions](https://wikipedia.org/wiki/Sigmoid_function). A 'Sigmoid Function' on a plot looks like an 'S' shape. It takes a value and maps it to somewhere between 0 and 1. Its curve is also called a 'logistic curve'. Its formula looks like thus:
>
> ![logi3](https://user-images.githubusercontent.com/78952426/126063661-af40a4fb-cd81-42b5-8eec-4b40ac681f01.png)
>
> where the sigmoid's midpoint finds itself at x's 0 point, L is the curve's maximum value, and k is the curve's steepness. If the outcome of the function is more than 0.5, the label in question will be given the class '1' of the binary choice. If not, it will be classified as '0'.
 
## Problem Statement
The data set contains Loan grades:7 classes where each class refers to digit(1:A,2:B,3:C,4:D,5:E,6:F,G:7).Objective of our model is to predict the correct grade ,based on given data and deploy that model to predict Loan Grade on Heroku using Flask.

## Build Multiclass Logistic Regression Model
Building a model to predict these multiclass is straightforward in Scikit-learn.
 - **Create Test and Train Dataset**

   select variables for classifiaction model and split dataset,so that we can use one set of data for training the model and one set for testing the model,split the training        and test sets calling `train_test_split()`:
   ```python
    from sklearn.model_selection import train_test_split
    
    features = [['funded_amnt_inv', 'int_rate', 'grade', 'emp_length','annual_inc', 'pymnt_plan', 'dti', 'deling_2yrs', 'fico_range_low','ing_last_6mths', 
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util','total_acc', 'initial_list_status', 'total_rec_late_fee','last_pymnt_amnt', 'last_fico_range_high',
    'last_fico_range_low','collections_12_mths_ex_med', 'policy_code', 'application_type','acc_now_dealing', 'tot_coll_amt', 'tot_cur_bal', 'total_bal_il',
     'max_bal_bc', 'issue_d_month', 'earliest_cr_line_month','earliest_cr_line_year', 'last_pymnt_d_year','last_credit_pull_d_month', 'last_credit_pull_d_year']]
    
   X=features.drop(['grade'],axis=1)
   y=features['grade']
    # split data into training and testing data, for both features and target
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.2, random_state=1)
    ```
  - **Normalization of independent features**
  
    we have normalize our data using using `StandardScaler()`:
    ```python
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.fit_transform(test_X)
    ```
    Now train the model, by calling `fit()` with training data:
    ```python
    from sklearn.metrics import accuracy_score,confusion_matrix, classification_report 
    from sklearn.linear_model import LogisticRegression
     accepted_model = LogisticRegression(penalty='none')
     accepted_model.fit(train_X, train_y)
     val_predictions = accepted_model.predict(test_X)
    
    ```
**Model Score**

Check the model score for training and test data:
 ```python
 print('Accuray: {}'.format(accepted_model.score(train_X,train_y)))
print("Accuracy: {}".format(accepted_model.score(test_X, test_y)))
```
```output
Accuray: 0.8697499145670882
Accuracy: 0.8702987103684982
```
**Confusion Matrix**
- Confusion Matrix helps us to visualize the performance of model.
- The diagonal elements represent the number of points for which the predicted label is equal to true label.
- Off-diagonal elements are those that are mislabeled by the classifier.
- The heigher the diagonal values of confusion matrix the better indicating many correct.
SO for our model confusion matrix is-
```python
confusion = confusion_matrix(test_y, val_predictions)
print(" \n Confusion Matrix is : \n", confusion)
```
```output
Confusion Matrix is : 
 [[ 75116   4178      0      0      0      0      0]
 [  5511 105556   7105      0      0      0      0]
 [    40   6860 103954   4483      0      0      0]
 [    27      0   6297  46486   3950      0      0]
 [     8      0      1   7558  14610   1211      0]
 [     6      0      0     11   3020   3746    503]
 [     1      0      0      0     87   1330    708]]
```
**Classification Report**

Classification report is used to measure the quality of prediction from classification algorithm
- Precision:Indiactes how many classes are correctly classified.
- Recall:Indicates what proportions of actual positives was identified correctly.
- F-score:It is the harmonic mean between precision & recall.
- Support:It is the number of occurences of the given class in our dataset.
```python
report = classification_report(test_y, val_predictions)
print(" \n Classification Report is : \n", report)
```
```output
Classification Report is : 
               precision    recall  f1-score   support

           1       0.93      0.95      0.94     79294
           2       0.91      0.89      0.90    118172
           3       0.88      0.90      0.89    115337
           4       0.79      0.82      0.81     56760
           5       0.66      0.61      0.63     23388
           6       0.55      0.46      0.50      7286
           7       0.57      0.33      0.42      2126

    accuracy                           0.87    402363
   macro avg       0.75      0.71      0.73    402363
weighted avg       0.86      0.87      0.87    402363
```

## Visualize the ROC curve of this model

This is not a bad model; its accuracy is in the 87% range so ideally you could use it to predict the loan grade given a set of variables.

Let's do one more visualization to see the so-called 'ROC' score:
```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# Get class probability scores
grade_prob = accepted_model.predict_proba(test_X)

# Get ROC metrics for each class
fpr = {}
tpr = {}
thresh ={}
for i in range(len(grade)):
  fpr[i], tpr[i], thresh[i] = roc_curve(test_y, grade_prob[:,i], pos_label=i) 
# Plot the ROC chart
plt.figure(figsize=(10,10))
plt.plot(fpr[0], tpr[0],color='orange', label=str(grade[0]) + ' vs Rest')
plt.plot(fpr[1], tpr[1],color='green', label=str(grade[1]) + ' vs Rest')
plt.plot(fpr[2], tpr[2],color='blue', label=str(grade[2]) + ' vs Rest')
plt.plot(fpr[3], tpr[3],color='red', label=str(grade[3]) + ' vs Rest')
plt.plot(fpr[4], tpr[4],color='yellow', label=str(grade[4]) + ' vs Rest')
plt.plot(fpr[5], tpr[5],color='pink', label=str(grade[5]) + ' vs Rest')
plt.plot(fpr[6], tpr[6],color='cyan', label=str(grade[6]) + ' vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
```
ROC curves are often used to get a view of the output of a classifier in terms of its true vs. false positives. "ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis."
![roc curve](https://user-images.githubusercontent.com/78952426/126062638-7c34bbb3-9bf7-44e5-9e3c-5fd0bcd411f9.png)
to compute the actual 'Area Under the Curve' (AUC):
```python
roc_auc = roc_auc_score(test_y, grade_prob, multi_class=('ovr'))
print(" \n AUC using ROC is : ", roc_auc)
```
```output
AUC using ROC is :  0.9865396293036989
```
The result is  0.9865396293036989. Given that the AUC ranges from 0 to 1, you want a big score, since a model that is 100% correct in its predictions will have an AUC of 1; in this case, the model is pretty good.

## **DEPLOYMENT AND WEB APPLICATION:**
  - What is mean by model deployment? Deploying a machine learning model, known as model deployment, simply means to integrate a machine learning model and integrate it into an existing production environment, where it can take in an input and return an output. 
  - The Lending Club’s Loan Grade Prediction Machine Learning Model is trained and tested using Logistic Regression with 87.02% accuracy. And using flask framework, HTML, CSS and python all the necessary files have created along with Procfile and requirement.txt. 
  - Using Heroku, Platform as a service we have successfully deployed our model in the online web server and provided ease to end users.
  - Deployed Online url:[http://loangradeprediction-api.herokuapp.com/](http://loangradeprediction-api.herokuapp.com/)
 

```output
- 1.	'funded_amnt_inv',  
- 2.	'int_rate',  
- 3.	'emp_length',  
- 4.	'annual_inc',  
- 5.	'pymnt_plan',  
- 6.	'dti',  
- 7.	'deling_2yrs',  
- 8.	'fico_range_low',  
- 9.	'ing_last_6mths',  
- 10.	'open_acc',  
- 11.	'pub_rec',  
- 12.	'revol_bal',  
- 13.	'revol_util',  
- 14.	'total_acc',  
- 15.	'initial_list_status',  
- 16.	'total_rec_late_fee',  
- 17.	'last_pymnt_amnt',  
- 18.	'last_fico_range_high',  
- 19.	'last_fico_range_low',  
- 20.	'collections_12_mths_ex_med',  
- 21.	'policy_code',  
- 22.	'application_type',  
- 23.	'acc_now_dealing',  
- 24.	'tot_coll_amt',  
- 25.	'tot_cur_bal',  
- 26.	'total_bal_il',  
- 27.	'max_bal_bc',  
- 28.	– 33. Datetime columns: 'issue_d_month', 'earliest_cr_line_month','earliest_cr_line_year','last_pymnt_d_month', 'last_credit_pull_d_month','last_credit_pull_d_year'__    
**Target value: ‘grade’**
```
```output
| File name      | Description | Description    |
| :---        |    :----:   |          ---: |
| accepted_2017_to_2018q4.csv      |    Information about loan accepted when they submit the application.    | 152   |
| rejected_2017_to_2018q4.csv   |    Information about loan rejected when they submit the application.     | 9      |  
```



