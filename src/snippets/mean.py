# Paste your values in this string
values = """
1
9
13
12
18
30
38
30
34
"""

# Split the string into a list of numbers, filter out empty strings, and convert to integers
numbers = [int(value) for value in values.split() if value.strip()]

# Calculate the mean
mean_value = sum(numbers) / len(numbers)

# Print the result
print(f"The mean of the values is: {mean_value}")
