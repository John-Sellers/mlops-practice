# Import the sys module to access system-specific parameters and functions
import sys

# Define a function named error_message_detail that takes two parameters: error and error_detail
def error_message_detail(error, error_detail):
    
    # Extract the traceback information from the error_detail using sys.exc_info()
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename where the error occurred from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Construct the error message using formatted strings (f-strings)
    error_message = f"Error occurred in python script: \n{file_name}, \n{exc_tb.tb_lineno}, \n{str(error)}"
    
    # Return the constructed error message
    return error_message
