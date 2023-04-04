#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# Is to identify products at risk of backorder before the event occurs so that business has time to react. 

# ## What is a Backorder?
# 
# * Backorders are products that are temporarily out of stock, but a customer is permitted to place an order against future inventory. 
# * A backorder generally indicates that customer demand for a product or service exceeds a company’s capacity to supply it. 
# * Back orders are both good and bad. Strong demand can drive back orders, but so can suboptimal planning. 

# ## Data description
# 
# Data file contains the historical data for the 8 weeks prior to the week we are trying to predict. The data was taken as weekly snapshots at the start of each week. Columns are defined as follows:
# 
#     sku - Random ID for the product
# 
#     national_inv - Current inventory level for the part
# 
#     lead_time - Transit time for product (if available)
# 
#     in_transit_qty - Amount of product in transit from source
# 
#     forecast_3_month - Forecast sales for the next 3 months
# 
#     forecast_6_month - Forecast sales for the next 6 months
# 
#     forecast_9_month - Forecast sales for the next 9 months
# 
#     sales_1_month - Sales quantity for the prior 1 month time period
# 
#     sales_3_month - Sales quantity for the prior 3 month time period
# 
#     sales_6_month - Sales quantity for the prior 6 month time period
# 
#     sales_9_month - Sales quantity for the prior 9 month time period
# 
#     min_bank - Minimum recommend amount to stock
# 
#     potential_issue - Source issue for part identified
# 
#     pieces_past_due - Parts overdue from source
# 
#     perf_6_month_avg - Source performance for prior 6 month period
# 
#     perf_12_month_avg - Source performance for prior 12 month period
# 
#     local_bo_qty - Amount of stock orders overdue
# 
#     deck_risk - Part risk flag
# 
#     oe_constraint - Part risk flag
# 
#     ppap_risk - Part risk flag
# 
#     stop_auto_buy - Part risk flag
# 
#     rev_stop - Part risk flag
# 
#     went_on_backorder - Product actually went on backorder. This is the target value.
#     
#          Yes or 1 : Product backordered
# 
#          No or 0  : Product not backordered

# #### Broad Classification of attributes
# 
# SKU: Unique material identifier;
# 
# INV: Current inventory level of material;
# 
# TIM: Registered transit time;
# 
# FOR-: Forecast sales for the next 3, 6, and 9 months;
# 
# SAL-: Sales quantity for the prior 1, 3, 5, and 9 months;
# 
# MIN: Minimum recommended amount in stock (MIN);
# 
# OVRP: Parts overdue from source;
# 
# SUP-: Supplier performance in last 1 and 2 semesters;
# 
# OVRA: Amount of stock orders overdue (OVRA);
# 
# RSK-: General risk flags associated to the material;
# 
# BO: Product went on backorder

# # Loading the required libraries

# In[ ]:


#!pip install graphviz
#!pip install imblearn


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

from imblearn.over_sampling import SMOTE

import graphviz


# # Identify Right Error Metrics
# 
#     Based on the business have to identify the right error metrics.

# ## Function to calculate required metrics

# In[ ]:


def evaluate_model(act, pred):
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))    


# # Loading the data

# In[ ]:


data = pd.read_csv("BackOrders.csv", header=0)


# # Understand the Data - - Exploratory Data Analysis (EDA)

# ## Number row and columns

# In[ ]:


data.shape


# ## First and last 5 records

# In[ ]:


data.head()


# In[ ]:


data.tail()


# ## Statistic summary 
#     Using describe function

# In[ ]:


data.describe(include='all')


# ## Data type

# In[ ]:


data.dtypes


# __Observations__
# 
# * sku is `categorical` but is interpreted as `int64` 
# * potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder are also `categorical` but is interpreted as `object`. 

# # Data pre-processing

# ## Convert all the attributes to appropriate type

# Data type conversion
# 
#     Using astype('category') to convert potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder attributes to categorical attributes.
# 

# In[ ]:


for col in ['sku', 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 
            'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
    data[col] = data[col].astype('category')


# ### Re-display data type of each variable

# In[ ]:


data.dtypes


# ### Statistic summary 
#     Using describe function

# In[ ]:


data.describe(include='all')


# ## Delete sku attribute

# In[ ]:


np.size(np.unique(data.sku, return_counts=True)[0])


# In[ ]:


data.drop('sku', axis=1, inplace=True)


# ## Missing Data
# 
#     Missing value analysis and dropping the records with missing values

# In[ ]:


data.isnull().sum()


# Observing the number of records before and after missing value records removal

# In[ ]:


print (data.shape)


# Since the number of missing values is about 5% and as we have around 61K records. For initial analysis we ignore all these records

# In[ ]:


data = data.dropna(axis=0)


# In[ ]:


print(data.isnull().sum())
print(data.shape)


# ## Encoding Categorical to Numeric
# 
# `pandas.get_dummies` To convert convert categorical variable into dummy/indicator variables

# __Creating dummy variables__
# 
# If we have k levels in a category, then we create k-1 dummy variables as the last one would be redundant. So we use the parameter drop_first in pd.get_dummies function that drops the first level in each of the category.

# In[ ]:


cat_attr_names = data.select_dtypes(include=['category']).columns


# In[ ]:


data = pd.get_dummies(columns=cat_attr_names, data=data, 
                      prefix=cat_attr_names, prefix_sep="_", drop_first=True)


# In[ ]:


print (data.columns, data.shape)


# ## Train and test split

# ### Target attribute distribution 

# In[ ]:


data['went_on_backorder_Yes'].value_counts()


# In[ ]:


data['went_on_backorder_Yes'].value_counts(normalize=True)*100


# ### Split the data into train and test
# 
# `sklearn.model_selection.train_test_split`
# 
#     Split arrays or matrices into random train and test subsets

# In[ ]:


X = data.drop('went_on_backorder_Yes', axis=1)
y = data['went_on_backorder_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123) 


# ### Target attribute distribution after the split

# In[ ]:


print(pd.value_counts(y_train, normalize=True) * 100)

print(pd.value_counts(y_test, normalize=True) * 100)


# # Model building

# ## Decision Tree Model

# ### Instantiate Model

# In[ ]:


dtclf = DecisionTreeClassifier()


# ### Train Model

# In[ ]:


dtclf.fit(X_train, y_train)


# ### List important features

# In[ ]:


importances = dtclf.feature_importances_
importances


# In[ ]:


indices = np.argsort(importances)[::-1]
ind_attr_names = X_train.columns
pd.DataFrame([ind_attr_names[indices], np.sort(importances)[::-1]])


# ### Display Tree

# In[ ]:


dtclf.classes_


# In[ ]:


dot_data = export_graphviz(dtclf, 
                           feature_names=ind_attr_names,
                           class_names=['No', 'Yes'], 
                           filled=True) 

graph = graphviz.Source(dot_data) 

graph.render("ClassTree") 


# In[ ]:


# Decision Tree Graph explanation
dtclf2 = DecisionTreeClassifier(max_depth=2) # Change 1, 2, 3
dtclf2.fit(X_train, y_train)
dot_data2 = export_graphviz(dtclf2, 
                           feature_names=ind_attr_names,
                           class_names=['No', 'Yes'], 
                           filled=True) 

graph2 = graphviz.Source(dot_data2) 
graph2


# ### Predict

# In[ ]:


train_pred = dtclf.predict(X_train)
test_pred = dtclf.predict(X_test)


# In[ ]:


def evaluate_model(act, pred):
    from sklearn.metrics import confusion_matrix, accuracy_score,     recall_score, precision_score
    
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))   


# ### Evaluate

# In[ ]:


print("--Train--")
evaluate_model(y_train, train_pred)
print("--Test--")
evaluate_model(y_test, test_pred)


# ### Observation:
#     Recall is pretty low
# 
# __Reason__:
# 
#     Class imbalance i.e. 81% of the records are not back order records 

# ## Up-sampling 
# 
#     Using SMOTE (Synthetic Minority Over-sampling Technique)

# ### Instantiate SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)


# ### Fit Sample

# In[ ]:


X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)


# In[ ]:


X_train_sm.shape


# In[ ]:


y_train_sm.shape


# In[ ]:


print(pd.value_counts(y_train_sm, normalize=True) * 100)


# ## Decision Tree with up-sample data

# ### Instantiate Model

# In[ ]:


dtclf2 = DecisionTreeClassifier()


# ### Train the model

# In[ ]:


dtclf2 = dtclf2.fit(X_train_sm, y_train_sm)


# ### List important features

# In[ ]:


importances = dtclf2.feature_importances_
importances


# In[ ]:


indices = np.argsort(importances)[::-1]
pd.DataFrame([ind_attr_names[indices], np.sort(importances)[::-1]])


# ### Predict

# In[ ]:


train_pred=dtclf2.predict(X_train_sm)
test_pred=dtclf2.predict(X_test)


# ### Evaluate

# In[ ]:


print("--Train--")
evaluate_model(y_train_sm, train_pred)
print("--Test--")
evaluate_model(y_test, test_pred)


# ## Hyper-parameter tuning using Grid Search and Cross Validation

# ### Cross validation
# 
# ![](grid_search_cross_validation.png)

# ### Two type of Machine Learning Model Parameters
# 
# In a machine learning model, there are 2 types of parameters:
# 
# * __`Model Parameters`__: These are the parameters that the model learn during training. These are also called fitted parameters.
# * __`Hyperparameters`__: These are adjustable parameters that must be tuned in order to obtain a model with optimal performance. These are fed into the model during training.
# 
# 
# 

# ### Lets list the hyper-parameters for Decision Trees:
# 
# 
# * criterion : string, optional (default=”gini”)
# 
#     The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
#     
# * min_samples_split : int, float, optional (default=2)
# 
#     The minimum number of samples required to split an internal node.
#    
# * max_depth : int or None, optional (default=None)
# 
#     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#     
# * min_samples_leaf : int, float, optional (default=1)
# 
#     The minimum number of samples required to be at a leaf node.

# ### Parameters to test

# In[ ]:


param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5],
              "max_depth": [4, 2],
              "min_samples_leaf": [1, 5]
             }


# ### Instantiate Decision Tree

# In[ ]:


dtclf3 = DecisionTreeClassifier()


# ### GridSearchCV 

# In[ ]:


from sklearn.model_selection import GridSearchCV
dtclf_grid = GridSearchCV(dtclf3, param_grid, cv=3)


# ### Train

# In[ ]:


dtclf_grid.fit(X_train_sm, y_train_sm)


# ### Best Params

# In[ ]:


dtclf_grid.best_params_


# ### Predict 

# In[ ]:


train_pred = dtclf_grid.predict(X_train_sm)
test_pred = dtclf_grid.predict(X_test)


# ### Evaluate

# In[ ]:


print("--Train--")
evaluate_model(y_train_sm, train_pred)
print("--Test--")
evaluate_model(y_test, test_pred)


# ## Building Decision Tree Model using Variable Importance

# In[ ]:


importances = dtclf_grid.best_estimator_.feature_importances_
importances


# In[ ]:


indices = np.argsort(importances)[::-1]
print(indices)


# In[ ]:


select = indices[0:5]
print(select)


# ### Instantiate Model

# In[ ]:


dtclf3 = DecisionTreeClassifier(criterion= 'entropy', 
                                max_depth= None, 
                                min_samples_leaf= 1,
                                min_samples_split= 2)


# ### Train the model

# In[ ]:


dtclf3 = dtclf3.fit(X_train_sm.values[:,select], y_train_sm)


# ### Predict

# In[ ]:


train_pred = dtclf3.predict(X_train_sm.values[:,select])
test_pred = dtclf3.predict(X_test.values[:,select])


# ### Evaluate

# In[ ]:


print("--Train--")
evaluate_model(y_train_sm, train_pred)
print("--Test--")
evaluate_model(y_test, test_pred)


# In[ ]:




