# Hotel-booking-status-predict

This repository has the datasets and scripts: train-validation-scoring(testing) in order to predict by means a classification algorithm about hotel bookings cancellations. 

The project is developed by using a classification algorithm: Logistic Regression

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
    ├── notebooks                   <- Jupyter notebooks. Naming convencion is a number (for ordering: N1, N2)
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

<p><small>Structure of Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
