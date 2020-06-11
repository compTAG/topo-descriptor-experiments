# http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip


import wget
import os
from zipfile import ZipFile

URL_MPEG7 = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
URL_EMNIST = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

dir_data = 'data'

dir_list = [
    dir_data,
    os.path.concat('graphs_005_approx','mnist'),
    'graphs_005_approx/mnist_imgs',
    'graphs_001_approx/mnist',
    'graphs_001_approx/mnist_imgs',
    'graphs_005_approx/mpeg7',
    'graphs_005_approx/mpeg7_imgs',
    'graphs_005_approx/mpeg7_extra',
    'graphs_001_approx/mpeg7',
    'graphs_001_approx/mpeg7_imgs',
    'graphs_001_approx/mpeg7_extra',
]


# Make directories so that generate_graphs runs properly
def make_folders(dir_list):
  for x in dir_list:
    make_folder(x)
    
def make_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Directory %s created" % folder_path)
  else:    
    print("Directory %s already exists" % folder_path)

    
def get_data(url, target_dir, data_set_name):
  # first check if we should do any downloading
  if os.path.exists(target_dir):
    print("Warning: directory %s alreday exists, not redownloading %s dataset" % target_dir, data_set_name)
    return

  data = wget.download(url)

  # Unzip data and store in data folder
  with ZipFile(mpeg7_data, 'r') as zip_ref:
    # TODO: python has a set of tools for creating tmp paths for file.
    # can we rename with extractall
    # can we get the name without assumeing "original"
    # when concating paths, use os.path.concat(...)
    tmp = 'tmp'
    zip_ref.extractall(tmp)

  # move
  dst = target_dir + data_set_name
  os.rename(tmp+'original', dst)
  print("Directory " , dst,  " Created ")
  
  # cleanup tmp  
  os.remove(tmp)

  return dst
    
def get_mpeg7(url, target_dir):
  dst = get_data(url, target_dir, 'mpeg7')
  
  # clean bad files
  os.remove(dst + 'rat-09.gif')
    

def get_emnist(url, target_dir):
  dst = get_data(url, target_dir, 'emnist')
    

def download_data():
  get_mpeg7(URL_MPEG7, data_dir)
  get_emnist(URL_EMNIST, data_dir)
