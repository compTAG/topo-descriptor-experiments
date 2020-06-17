import wget
import os
from backports import tempfile
from zipfile import ZipFile

URL_MPEG7 = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
URL_EMNIST = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

dir_data = 'data'

dir_list = [
    dir_data,
    os.path.join('graphs_005_approx','mnist'),
    os.path.join('graphs_005_approx','mnist_imgs'),
    os.path.join('graphs_001_approx', 'mnist'),
    os.path.join('graphs_001_approx', 'mnist_imgs'),
    os.path.join('graphs_005_approx','mpeg7'),
    os.path.join('graphs_005_approx', 'mpeg7_imgs'),
    os.path.join('graphs_005_approx', 'mpeg7_extra'),
    os.path.join('graphs_001_approx', 'mpeg7'),
    os.path.join('graphs_001_approx', 'mpeg7_imgs'),
    os.path.join('graphs_001_approx', 'mpeg7_extra'),
    os.path.join('graphs', 'random_imgs'),
    os.path.join('graphs', 'random'),
]


# Make directories so that generate_graphs runs properly
def make_folders(dir_list):
  for path in dir_list:
    make_folder(path)
    
def make_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Directory %s created" % folder_path)
  else:    
    print("Directory %s already exists" % folder_path)

    
def get_data(url, target_dir, data_set_name):
  # first check if we should do any downloading
  if os.path.exists(os.path.join(target_dir, data_set_name)):
    print("Warning: directory %s already exists, not re-downloading %s dataset" % (target_dir, data_set_name))
    return

  with tempfile.TemporaryDirectory() as tmp:
    data = wget.download(url,tmp)

    # Unzip data and store in data folder
    with ZipFile(data, 'r') as zip_ref:
      zip_ref.extractall(tmp)
      temp_dir = os.path.dirname(zip_ref.namelist()[1])
      
      # move and rename
      dst = os.path.join(target_dir, data_set_name)
      os.rename(os.path.join(tmp,temp_dir), dst)
      print("Directory " , dst,  " Created ")

  return dst
    
def get_mpeg7(url, target_dir):
  dst = get_data(url, target_dir, 'mpeg7')
  
  # clean bad files
  if os.path.exists('mpeg7/rat-09.gif'):
    os.remove('mpeg7/rat-09.gif')
  

def get_emnist(url, target_dir):
  dst = get_data(url, target_dir, 'emnist')
    

def download_data():
  get_mpeg7(URL_MPEG7, dir_data)
  get_emnist(URL_EMNIST, dir_data)

def preprocess_data(dir_list):
    make_folders(dir_list)
    download_data()
