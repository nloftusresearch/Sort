# Read the raw text from the file
with open(r'D:\Coding\Thesis\sort\object_dict.txt', 'r') as file:
    raw_text = file.read()

# Find the part of the text that represents the dictionary (remove variable name)
start_index = raw_text.find('{')
end_index = raw_text.rfind('}')
dictionary_text = raw_text[start_index:end_index + 1]

# Use eval to convert the dictionary text into a dictionary
color_mapping = eval(dictionary_text)

# Now color_mapping is a dictionary
print(color_mapping)
