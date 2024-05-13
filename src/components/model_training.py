import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:

            logging.info("Splitting training and testing input data")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Grandient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBoost Classifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train, X_test, y_train, y_test, models
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.65:
                raise CustomException("No best model found")

            logging.info("Found best model based on training and testing datasets")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
