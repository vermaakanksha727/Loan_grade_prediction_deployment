# Logistic Regession to Predict Loan Grade
**Logistic Regression** is a Machine Learning classification algorithnm that is used to predict the probability of categorical dependent variable.In logistic regression,the dependent variable is a **binary** or **Multinomial**, which involves having more than one category.

# Logistic Regression Assumptions:
- The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
- The independent variables are linearly related to the log odds.
- Logistic regression requires quite large sample sizes.

> **🧮 Mathematics behind logistic regression**
>
> Remember how linear regression often used ordinary least squares to arrive at a value? Logistic regression relies on the concept of 'maximum likelihood' using [sigmoid functions](https://wikipedia.org/wiki/Sigmoid_function). A 'Sigmoid Function' on a plot looks like an 'S' shape. It takes a value and maps it to somewhere between 0 and 1. Its curve is also called a 'logistic curve'. Its formula looks like thus:
>
> ![logistic function](images/sigmoid.png)
>
> where the sigmoid's midpoint finds itself at x's 0 point, L is the curve's maximum value, and k is the curve's steepness. If the outcome of the function is more than 0.5, the label in question will be given the class '1' of the binary choice. If not, it will be classified as '0'.
 
## Problem Statement
The data set contains Loan grades:7 classes where each class refers to digit(1:A,2:B,3:C,4:D,5:E,6:F,G:7).Objective of our model is to predict the correct grade ,based on given data and deploy that model to predict Loan Grade on Heroku using Flask.

## Build Multiclass Logistic Regression Model
Building a model to find these Multi classification is straightforward in Scikit-learn.
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
    Now train the model, by calling `fit()` with training data, and print out its result:
    ```python
    from sklearn.metrics import accuracy_score, classification_report 
    from sklearn.linear_model import LogisticRegression
     accepted_model = LogisticRegression(penalty='none')
     accepted_model.fit(train_X, train_y)
     val_predictions = accepted_model.predict(test_X)
    
    print(classification_report(test_y,val_predictions))
    print('Accuracy: ', accuracy_score(test_y, predictions))
    ```

