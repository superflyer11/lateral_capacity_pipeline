import os
import re

# Directory containing the files
directory = "/mofem_install/jupyter/thomas/mfront_example_test/simulations/pile_day_96_sim_18_20241121_150113_vM/vtks"

# List all files in the directory
files = os.listdir(directory)

# Extract numbers from filenames
numbers = sorted([int(re.search(r'\d+', file).group()) for file in files if re.search(r'\d+', file)])

# Find missing numbers
missing_numbers = [num for num in range(numbers[0], numbers[-1] + 1) if num not in numbers]

print("Missing numbers:", missing_numbers)