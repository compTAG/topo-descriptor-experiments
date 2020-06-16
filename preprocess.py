from get_data import *
from tqdm import tqdm 


# Should this step also generate the output folders for running experiments? 


# Generate appropriate folders for data processing and download data.
def preprocess_data(dir_list):
  make_folders(dir_list)
  download_data()

preprocess_data(dir_list)



