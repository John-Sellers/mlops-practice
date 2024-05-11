import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Import custom exception and logger
from src.exception import CustomException
from src.logger import logging

# Define a data class for data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Define a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Load data from CSV file into a DataFrame
            df = pd.read_csv(r'C:\User\selle\Downloads\code_dev\mlops-practice\notebook\data\StudentsPerformance.csv')
            
            # Log information about starting data ingestion
            logging.info('Data ingestion has commenced')

            # Create directories if they don't exist for saving data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Log information about starting train-test split
            logging.info('Train test split has commenced')

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)

            # Save the train and test sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Log information about successful completion of data ingestion
            logging.info('Ingestion has been finished and completed')

            # Return paths to the train and test data files
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            # Log an error message and raise a custom exception
            logging.info('An error has occurred!')
            raise CustomException(e, sys)
