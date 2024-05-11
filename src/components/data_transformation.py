import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a preprocessor object using ColumnTransformer
        """
        try:
            # Define numerical and categorical columns
            numerical_cols = ['reading score', 'writing score']
            categorical_cols = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            # Define pipelines for numerical and categorical transformations
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            # Log categorical and numerical columns
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # Combine numerical and categorical pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info('An error has occurred creating the data transformer object')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation using the preprocessor object
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test datasets')

            logging.info('Preparing for preprocessing')

            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_col = 'math score'
            numerical_cols = ['writing score', 'reading score']

            # Split data into input and target features
            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info(f'Preprocessing is being applied')

            # Fit and transform input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features with target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info('An error has occurred initiating preprocessing')
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example usage
    obj = DataTransformation()
    train_data, test_data = obj.initiate_data_transformation('train.csv', 'test.csv')
