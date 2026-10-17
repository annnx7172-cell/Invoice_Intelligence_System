import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
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
    
CV_SPLITS = 5
RANDOM_STATE = 42


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Tunes every regression model with GridSearchCV (scoring = R2) on the TRAINING
    data only.

    LEAKAGE FIX: models used to be ranked on the test set, i.e. the test set picked
    the winner and was then also used to report the winner's score. Now models are
    ranked on cross-validated R2 computed inside the training set; the test set is
    only scored for reporting.

    Returns (cv_report, test_report): both {model_name: R2}.
    """
    try:
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_report, test_report = {}, {}

        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=cv, scoring="r2")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            cv_report[name] = gs.best_score_
            test_report[name] = r2_score(y_test, model.predict(X_test))

        return cv_report, test_report

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