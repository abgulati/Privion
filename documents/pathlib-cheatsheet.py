from pathlib import Path

'''
Benefits of Using pathlib:
Readability: The use of the / operator for joining paths makes the code more readable and intuitive, resembling natural directory structures.

Cross-Platform Compatibility: pathlib automatically handles different path separators for different operating systems, making your code more portable.

Object-Oriented Approach: pathlib provides an object-oriented interface, which can lead to cleaner and more maintainable code, especially when performing multiple operations on the same path.

Consistency: Using pathlib throughout your codebase can lead to more consistent path handling, reducing the likelihood of errors related to path manipulation.

Additional Features: pathlib offers a wide range of methods for path manipulation and file system operations, which can simplify your code further.

Overall, refactoring to use pathlib can enhance the clarity and robustness of your code, especially in projects where path manipulation is frequent.
'''

from pathlib import Path
import os

# Define a path using pathlib
path_str = r'C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs'
path = Path(path_str)
# output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs')

# Convert to string for storage
path_str = str(path)  # or path.as_posix()
# output: 'C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs'

# Convert back to Path object
path = Path(path_str)
# output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs')

# Normalize the path
normalized_path = path.resolve()

# Print the normalized path
print("Normalized Path:", normalized_path)
# output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs')

# pathlib equivalent to `os.makedirs(str(path), exist_ok=True)`
path.mkdir(parents=True, exist_ok=True) # parents=True allows the creation of parent directories as needed, similar to the recursive behavior of os.makedirs().

# pathlib equivalent to `os.getcwd()`
print("Current Working Directory:", Path.cwd()) # output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs')

# Join paths in a platform-agnostic way
full_path = normalized_path / 'test.txt'

# Print the full path
print("Full Path:", full_path)
# output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs\test.txt')

# Remove the file using os.remove()
os.remove(str(full_path))

# Access parts of the path
print("Parent Directory:", full_path.parent)    # output: WindowsPath('C:\Users\abhee\OneDrive\Documents\Work_Stuff\LLMs\LARS-Enterprise\documents\sample_docs')
print("File Name:", full_path.name)            # output: test.txt
print("File Stem:", full_path.stem)            # output: test
print("File Suffix:", full_path.suffix)        # output: .txt

# List directory contents
if path.is_dir():
    print("Directory Contents:", list(path.iterdir())) # output: ['test.txt']

# Check if the path exists
print("Path Exists:", path.exists()) # output: True

# Check if file exists
print("File Exists:", full_path.exists()) # output: True

# Get the size of the file
# To get the size of a file, you can use the .stat() method, which returns a os.stat_result object containing various statistics about the file. You can then access the st_size attribute to get the file size:

if full_path.is_file() and full_path.stat().st_size > 0:     # .is_file(): Ensures that the path is a file, not a directory.
    print("The file exists and is not empty.")
else:
    print("The file does not exist or is empty.")