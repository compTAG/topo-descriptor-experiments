from get_data import *
from generate_graphs import *
from tqdm import tqdm 

#def preprocess_data(directories):
 # make_folders(directories)
  #download_data()
  #main()


make_folders(dir_list)
get_mpeg7(URL_MPEG7, dir_data)
get_emnist(URL_EMNIST, dir_data)
main()

