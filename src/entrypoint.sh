#!/bin/bash

# Start the SageMaker inference server using sagemaker-inference-containers
exec python -m sagemaker_inference.entry_point serve