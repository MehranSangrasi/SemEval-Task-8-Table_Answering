file_path = "data/databench.txt"

# Open the file and count the lines
with open(file_path, "r") as file:
    line_count = sum(1 for line in file)

print(f"Number of rows in the text file: {line_count}")