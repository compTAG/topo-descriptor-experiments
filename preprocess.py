from get_data import *
from generate_graphs import *

def preprocess_data(directories):
  make_folders(directories)
  download_data()
  main()