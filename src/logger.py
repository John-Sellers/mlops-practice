import logging
import os
from datetime import datetime

# Generating a log file name based on the current date and time
log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Creating a path for the log file in a 'logs' directory within the current working directory
logs_path = os.path.join(os.getcwd(), "logs", log_file)

# Creating the 'logs' directory (if it doesn't exist) to store log files
# exist_ok=True ensures no error is raised if the directory already exists
os.makedirs(logs_path, exist_ok=True)

# Creating the full path to the log file by joining the log directory path and the log file name
log_file_path = os.path.join(logs_path, log_file)

# Configuring the logging system with the specified parameters
logging.basicConfig(
    # Specifying the file where log messages will be written (the dynamically generated log file)
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

if __name__ == '__main__':
    logging.info('Logging has started')