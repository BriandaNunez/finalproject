import os

class SavedFile:
    def __init__(self, file_name):
        self.file_name = file_name

    def check_saved(self):
        if os.path.exists(self.file_name):
            print("The file has been saved successfully.")
        else:
            print("The file could not be found. It may not have been saved correctly.")

# Create an instance of the SavedFile class
file = SavedFile(r'Project\dtmodel.pkl')

# Check if the file has been saved correctly
file.check_saved()
