import json
import ast

file_path = '/remote_home/Thesis/Sort/frames_output.txt'

# Read data from the file
with open(file_path, 'r') as file:
    data = file.read()


print(data[0:])

'''data_dict = ast.literal_eval(data)

# Now data_dict is a dictionary
print(data_dict[0:])'''
