#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Script for training model: Logistic Regression
#**********************************************
import numpy as np
import pandas as pd
import os
import pickle

# To build linear model for statistical analysis and prediction
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,    
)

# let's check the VIF of the predictors
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# In[50]:


# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    This is for computing different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold #if is more than threshold then print
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[51]:


def checking_vif(predictors):    
    vif_series1 = pd.Series(
        [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])],
        index=predictors.columns,
    )
    i=0
    for num in vif_series1: 
        num='{0:.4g}'.format(num)
        vif_series1[i]=num
        i=i+1
    return vif_series1[vif_series1.values<=10], vif_series1[vif_series1.values>10]        


# In[52]:


def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed/', filename))
    df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
    return df


# In[53]:


def training(filename):
    df=read_file_csv(filename)
    X_train=df.drop(labels=['booking_status'],axis=1)
    y_train=df['booking_status']
    
    # fitting the model on training set
    logit = sm.Logit(y_train, X_train.astype(float))
    lg = logit.fit(method='bfgs')

    #printing training performance
    print("Training Performance:")
    model_performance_classification_statsmodels(lg, X_train, y_train)
    
    #checking VIF of the predictors
    vif_less10,vif_greater10 = checking_vif(X_train)    

    #Dropping first variable
    X_train1 = X_train.drop("market_segment_type_Online", axis=1)
    
    #Checking VIF
    vif_less10,vif_greater10 = checking_vif(X_train1)

    #fitting the model on training set and printing performance again
    logit1 = sm.Logit(y_train, X_train1.astype(float))
    lg1 = logit1.fit(method='bfgs')
    print("Training Performance:")
    model_performance_classification_statsmodels(lg1, X_train1, y_train)

    #Dropping second variable    
    X_train2 = X_train1.drop("no_of_week_nights_log", axis=1)
    
    #Checking VIF
    vif_less10,vif_greater10 = checking_vif(X_train2)
    
    #fitting the model on training set and printing performance again
    logit2 = sm.Logit(y_train, X_train2.astype(float))
    lg2 = logit2.fit(method='bfgs')
    print("Training Performance:")
    model_performance_classification_statsmodels(lg2, X_train2, y_train)
    
    #Dropping high p-values
    cols = X_train2.columns.tolist()

    # setting an initial max p-value
    max_p_value = 1

    while len(cols) > 0:
        # defining the train set
        X_train_aux = X_train2[cols]

        # fitting the model
        model = sm.Logit(y_train, X_train_aux).fit(disp=False, method='bfgs')

        # getting the p-values and the maximum p-value
        p_values = model.pvalues
        max_p_value = max(p_values)

        # name of the variable with maximum p-value
        feature_with_p_max = p_values.idxmax()

        if max_p_value > 0.05:
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features = cols
    print(selected_features)
          
    #Final training of model
    X_train3 = X_train2[selected_features]
    logit3 = sm.Logit(y_train, X_train3.astype(float))
    lg3 = logit3.fit(method='bfgs')
    print(lg3.summary())
          
    #saving the model to use into production later
    filename = '../models/best_model.pkl'
    pickle.dump(lg3, open(filename, 'wb'))
    
    #saving the new features selected for the model      
    pd.Series(selected_features).to_csv("../data/processed/selected_features.csv")          


# In[54]:


#training from main function
def main():
    training('booking_train.csv')
    print('The training of model was ended')


# In[55]:


if __name__ == "__main__":
    main()

