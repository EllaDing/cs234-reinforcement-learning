import os
import re

# Directory containing the files
directory = "."

# Regular expression pattern to match the code block with variable '#' counts
pattern = re.compile(r"(^\s*#+\s*YOUR CODE HERE.*?\n)(.*?)(^\s*#+\s*END YOUR CODE.*?$)", re.DOTALL | re.MULTILINE)

# Process each file in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    if os.path.isfile(filepath):  # Ensure it's a file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace matched blocks with only the surrounding comments
        new_content = re.sub(pattern, r"\1\3", content)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

print("Code removal complete.")
