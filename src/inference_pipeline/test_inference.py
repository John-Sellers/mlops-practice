import sys

import os
import json
import pandas as pd

current_dir = os.path.abspath('')
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from inference import model_fn, predict_fn, CustomData

# Define the model directory (replace with your actual model directory)
model_dir = "artifacts/"

# Load the model and preprocessor
model, preprocessor = model_fn(model_dir)

# Create a sample input using the CustomData class
sample_input = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74,
)

# Convert the sample input to a DataFrame
input_df = sample_input.get_data_as_dataframe()

# Predict using the loaded model and preprocessor
prediction = predict_fn(input_df, (model, preprocessor))

# Print the result
print(f"Prediction: {prediction}")