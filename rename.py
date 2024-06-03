import os

directory = os.getcwd()  # Get the current directory
files = os.listdir(directory)  # List all files in the directory

index = 0
for filename in files:
    if (filename != "rename.py"):
        newname = f"{index}.wav"
        os.rename(filename, newname)
        index +=1