import urllib.request as request
import os
from CNN_Classifier.entity.config_entity import DataIngestionConfig
import zipfile
from pathlib import Path
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config= config
    def download_file(self):
        if not os.path.exists(self.config.local_data_path):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_path
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_path))}")  

    def extract_zip_file(self):
            """
            zip_file_path: str
            Extracts the zip file into the data directory
            Function returns None
            """
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)        

