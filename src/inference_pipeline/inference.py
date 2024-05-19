import os
import sys
import pandas as pd

sys.path.append(".")
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, features):
        try:
            prep_data = self.preprocessor.transform(features)
            pred = self.model.predict(prep_data)
            return pred
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            input_data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(input_data)
        except Exception as e:
            raise CustomException(e, sys)


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    model = load_object(file_path=model_path)
    preprocessor = load_object(file_path=preprocessor_path)
    return model, preprocessor


def predict_fn(input_data, model):
    model, preprocessor = model
    pipeline = PredictPipeline(model, preprocessor)
    prediction = pipeline.predict(input_data)
    return prediction