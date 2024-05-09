import sys

import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occurred in python script: \n{file_name}, \n{exc_tb.tb_lineno}, \n{str(error)}"
    
    return error_message