import sys
import time
from typing import Callable, Any
from functools import wraps

def print_progress(iteration, total, decimals=1, bar_length=50):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r |%s| %s%s (%s of %s)' %
                     (bar, percents, '%', str(iteration), str(total)))
    sys.stdout.flush()
   
def is_not_h5m(file):
    return not file.endswith('h5m')

# @track_time("RUNNING COMMAND...")
def run_command(index, command, log_file):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            log_file.write(output)
            log_file.flush()
            sys.stdout.write(output)
            sys.stdout.flush()
    
    # Capture remaining output after process ends
    for output in process.stdout.readlines():
        log_file.write(output)
        log_file.flush()
        sys.stdout.write(output)
        sys.stdout.flush()
    
    return process

def print_message_in_box(message: str) -> Callable:
    """
    A decorator that prints a message within a box of ASCII characters before executing the function.
    
    Args:
        message (str): The message to be printed.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Calculate the length of the message and adjust the border length accordingly
            message_length = len(message)
            border_length = message_length + 4  # Adjust for padding and borders
            top_border = "╭" + "─" * border_length + "╮"
            middle_line = f"│  {message}  │"
            bottom_border = "╰" + "─" * border_length + "╯"
            
            # Print the formatted message
            print(top_border)
            print(middle_line)
            print(bottom_border)
            
            # Execute the wrapped function
            return func(*args, **kwargs)
        return wrapper
    return decorator

def print_message_in_box_plain(message: str) -> Callable:
    message_length = len(message)
    border_length = message_length + 4  # Adjust for padding and borders
    top_border = "╭" + "─" * border_length + "╮"
    middle_line = f"│  {message}  │"
    bottom_border = "╰" + "─" * border_length + "╯"
    
    # Print the formatted message
    print(top_border)
    print(middle_line)
    print(bottom_border)

def track_time(message: str) -> Callable:
    """
    Decorator that prints a message and tracks the execution time of a function.
    
    Args:
        message (str): The message to be printed before execution.
        
    Returns:
        Callable: The wrapped function with added time tracking.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Calculate the length of the message and adjust the border length accordingly
            message_length = len(message)
            border_length = message_length + 4  # Adjust for padding and borders
            top_border = "╭" + "─" * border_length + "╮"
            middle_line = f"│  {message}  │"
            bottom_border = "╰" + "─" * border_length + "╯"
            
            # Print the formatted message
            print(top_border)
            print(middle_line)
            print(bottom_border)
            
            start_wall_time = time.time()
            start_cpu_time = time.process_time()
            result = func(*args, **kwargs)
            elapsed_wall_time = time.time() - start_wall_time
            elapsed_cpu_time = time.process_time() - start_cpu_time
            print(f"\nDone, taken Wall Time: {elapsed_wall_time:.2f} seconds, CPU Time: {elapsed_cpu_time:.2f} seconds")
            return result
        return wrapper
    return decorator

