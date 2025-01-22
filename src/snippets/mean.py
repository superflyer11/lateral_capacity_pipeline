# Paste your values in this string
values = """
2
10
10
12
9
8
10
11
10
17
20
15
16
31
20
34
29
34
37
32
63
75
127
86
118
66
54
45
63
109
85
90
98
"""

# Split the string into a list of numbers, filter out empty strings, and convert to integers
numbers = [int(value) for value in values.split() if value.strip()]

# Calculate the mean
mean_value = sum(numbers) / len(numbers)

# Print the result
print(f"The mean of the values is: {mean_value}")
