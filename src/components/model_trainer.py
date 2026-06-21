import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    VotingRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import r2_score, f1_score
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, evaluate_classification_models


@dataclass
class ModelTrainerConfig:
    freight_model_file_path = os.path.join("artifacts", "freight_model.pkl")
    risk_model_file_path = os.path.join("artifacts", "risk_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # =====================================================================
    # FREIGHT (REGRESSION)
    # =====================================================================

    def train_freight_model(self, train_array, test_array):
        try:
            logging.info("Splitting features and target for Freight regression")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            regression_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(oob_score=True),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "SVR": SVR(),
            }

            regression_params = {
                "Linear Regression": {},
                "Ridge Regression": {'alpha': [0.01, 0.1, 1, 10]},
                "Lasso Regression": {'alpha': [0.01, 0.1, 1, 10]},
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse']},
                "Random Forest": {'n_estimators': [50, 100, 200]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },
                "SVR": {
                    'kernel': ['rbf', 'linear'],
                    'C': [1, 10, 100]
                },
            }

            freight_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=regression_models, param=regression_params
            )

            voting_regressor = VotingRegressor(estimators=[
                ('rf', regression_models["Random Forest"]),
                ('gb', regression_models["Gradient Boosting"]),
                ('lasso', regression_models["Lasso Regression"])
            ])
            voting_regressor.fit(X_train, y_train)
            voting_pred = voting_regressor.predict(X_test)
            freight_report["Voting Regressor"] = r2_score(y_test, voting_pred)
            regression_models["Voting Regressor"] = voting_regressor

            logging.info(f"Freight Model Report: {freight_report}")
            print("\n===== Freight Regression - R² Scores =====")
            for name, score in freight_report.items():
                print(f"{name}: {score:.4f}")
            print("============================================\n")

            rf_model = regression_models["Random Forest"]
            if hasattr(rf_model, 'oob_score_'):
                print(f"Random Forest OOB Score (Freight): {rf_model.oob_score_:.4f}\n")

            best_score = max(freight_report.values())
            best_name = max(freight_report, key=freight_report.get)
            best_model = regression_models[best_name]

            logging.info(f"Best freight model: {best_name} with R² {best_score}")

            save_object(
                file_path=self.model_trainer_config.freight_model_file_path,
                obj=best_model
            )

            return best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # RISK (CLASSIFICATION)
    # =====================================================================

    def train_risk_model(self, train_array, test_array):
        try:
            logging.info("Splitting features and target for Risk classification")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            classification_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(oob_score=True),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVC": SVC(probability=True),
            }

            classification_params = {
                "Logistic Regression": {'C': [0.1, 1, 10]},
                "Decision Tree": {'criterion': ['gini', 'entropy']},
                "Random Forest": {'n_estimators': [50, 100, 200]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },
                "SVC": {
                    'kernel': ['rbf', 'linear'],
                    'C': [1, 10, 100]
                },
            }

            risk_report = evaluate_classification_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=classification_models, param=classification_params
            )

            voting_classifier = VotingClassifier(estimators=[
                ('rf', classification_models["Random Forest"]),
                ('gb', classification_models["Gradient Boosting"]),
                ('lr', classification_models["Logistic Regression"])
            ], voting='soft')
            voting_classifier.fit(X_train, y_train)
            voting_pred = voting_classifier.predict(X_test)
            risk_report["Voting Classifier"] = f1_score(y_test, voting_pred)
            classification_models["Voting Classifier"] = voting_classifier

            logging.info(f"Risk Model Report: {risk_report}")
            print("\n===== Invoice Risk Classification - F1 Scores =====")
            for name, score in risk_report.items():
                print(f"{name}: {score:.4f}")
            print("=====================================================\n")

            rf_clf_model = classification_models["Random Forest"]
            if hasattr(rf_clf_model, 'oob_score_'):
                print(f"Random Forest OOB Score (Risk): {rf_clf_model.oob_score_:.4f}\n")

            best_score = max(risk_report.values())
            best_name = max(risk_report, key=risk_report.get)
            best_model = classification_models[best_name]

            logging.info(f"Best risk model: {best_name} with F1 {best_score}")

            save_object(
                file_path=self.model_trainer_config.risk_model_file_path,
                obj=best_model
            )

            return best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)