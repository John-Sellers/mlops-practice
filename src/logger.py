import logging
import os
from datetime import datetime

# Generating a log file name based on the current date and time
log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Creating the 'logs' directory (if it doesn't exist) to store log files
# exist_ok=True ensures no error is raised if the directory already exists
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Creating the full path to the log file within the 'logs' directory
log_file_path = os.path.join(logs_path, log_file)

# Configuring the logging system with the specified parameters
logging.basicConfig(
    # Specifying the file where log messages will be written (inside the 'logs' directory)
    filename=log_file_path,
    
    # Defining the format of log messages using placeholders
    # %(asctime)s: Timestamp of the log message
    # %(lineno)d: Line number where the logging call was made
    # %(name)s: Logger name
    # %(levelname)s: Log level (e.g., INFO, WARNING, ERROR)
    # %(message)s: Log message content
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    
    # Setting the logging level to INFO (only INFO and higher-level messages will be logged)
    level=logging.INFO
)
