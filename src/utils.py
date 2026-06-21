import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingRegressor

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def get_voting_regressor_score(X_train, y_train, X_test, y_test, models):
    try:
        voting_model = VotingRegressor(estimators=[
            ('rf', models["Random Forest"]),
            ('xgb', models["XGBRegressor"]),
            ('cat', models["CatBoosting Regressor"])
        ])
        voting_model.fit(X_train, y_train)
        voting_pred = voting_model.predict(X_test)
        voting_score = r2_score(y_test, voting_pred)

        return voting_model, voting_score

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    """
    Same pattern as evaluate_models, but for classification.
    Returns both the F1 score report AND a confusion matrix breakdown
    (TP, TN, FP, FN) for every model, so you can compare not just
    overall performance but the specific TYPE of errors each model makes.
    """
    try:
        report = {}
        confusion_details = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = f1_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
            confusion_details[list(models.keys())[i]] = {
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
            }

        return report, confusion_details

    except Exception as e:
        raise CustomException(e, sys)