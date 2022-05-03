import os
import zipfile

class Data_Recup:
    """
    Get the data of birds from kaggle 
    """

    def __init__(self, file_key_path : str):
        self.file_key_path = file_key_path

    def get_songs(self):

        os.system('mkdir ~/.kaggle')
        os.system('cp ' + self.file_key_path + ' ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        os.system('kaggle datasets download -d monogenea/birdsongs-from-europe')
        
        with zipfile.ZipFile('birdsongs-from-europe.zip', 'r') as zip_ref:
            zip_ref.extractall(".")
