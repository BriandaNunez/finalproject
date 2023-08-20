import logging
import os

# Construct the full log file path
log_file = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\tests\modeltest.log'

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO)

# Create a logger for 'modeltest'
logger = logging.getLogger('modeltest')

class SavedFile:
    def __init__(self, file_name):
        self.file_name = file_name

    def check_saved(self):
        if os.path.exists(self.file_name):
            logger.info("The file has been saved successfully.")
        else:
            logger.info("The file could not be found. It may not have been saved correctly.")

def test_saved_file():
    # Create an instance of the SavedFile class
    file = SavedFile(r'Project\finalproject_itesm_mlops\models\dtmodel.pkl')

    # Check if the file has been saved correctly
    file.check_saved()

# Run the test
def main():
    test_saved_file()

if __name__ == '__main__':
    main()
