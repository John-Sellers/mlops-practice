import os
import sys
from dataclasses import dataclass

# Importing machine learning models and evaluation metrics
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

# Importing custom exceptions, logger, and utility functions
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


# Define a data class for model trainer configuration
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# Define a class for model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            # Log information about splitting training and testing data
            logging.info("Splitting training and testing input data")

            # Extract features and target variables from the training and testing arrays
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1],
            )

            # Define a dictionary of machine learning models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBoost Classifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            # Evaluate the models and get a report of their performance
            model_report: dict = evaluate_model(
                X_train, X_test, y_train, y_test, models
            )

            # Find the best performing model based on the evaluation report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # If the best model's performance is below a threshold, raise an exception
            if best_model_score < 0.65:
                raise CustomException("No best model found")

            # Log information about the best model found
            logging.info("Found best model based on training and testing datasets")

            # Save the best model to a file
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # Make predictions using the best model on the testing data
            predicted = best_model.predict(X_test)

            # Calculate and return the R-squared score of the best model
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            # If any exception occurs, raise a custom exception with the error message
            raise CustomException(e, sys)
