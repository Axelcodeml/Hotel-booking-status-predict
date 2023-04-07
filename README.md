# Hotel-booking-status-predict
## The project was developed by using a classification algorithm: Logistic Regression
### This repository has the scripts: train-validation-scoring(testing) in order to predict with a classification algorithm about hotel bookings cancellations. 


Project Organization
----------------------


    ├── README.md                   <- The Top-level README for developers using this project
    ├── data
    |   ├── scores                  <- Results from scoring model
    |   ├── processed               <- Canonical datasets with selected feature after processing original dataset
    |   ├── raw                     <- Original dataset, inmutable data dump
    |
    ├── model                       <- Trained and serialized models, model predictions, or model summaries
    |
    ├── notebooks                   <- Jupyter notebooks. Naming convencion is a number 
    |                                    (for ordering: N1, N2)
    |
    ├── requirements.txt
    |
    |
    ├── src
    |   ├── __init__.py
    |   |
    |   ├── make_dataset.py         <- Script to prepare data
    |   |
    |   ├── train.py                <- Script to train models
    |   |
    |   ├── evaluate.py             <- Script to evaluate models using kpi's
    |   |
    │   └── predict.py              <- Script to use trained models to make predictions
    |
    └── LICENSE                     <- License
    
    --------

#### The dataset was provided during my data science training where classification algorithm like Logistic Regression and Decistion Tree wee applied.

To apply Logistic Regression, we have needed to drop-out some variables to determine what variables are necessary to process the modeling. We have also considered the p-value parameter like other way to drop-out the unnecessary variables. These variable pre-processing you can see in **Preparing_data.ipynb** from Notebook file.

<p><small>Structure of Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
