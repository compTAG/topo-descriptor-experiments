import wget
import os
from backports import tempfile
from zipfile import ZipFile
import shutil

URL_MPEG7 = 'https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip'
URL_EMNIST = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
URL_MAP_CONSTRUCTION = 'https://github.com/pfoser/mapconstruction/zipball/master'

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
    os.path.join('output_001_approx', 'smallest_angle_exp', 'mnist'),
    os.path.join('output_001_approx', 'smallest_angle_exp', 'mpeg7'),
    os.path.join('output_001_approx', 'smallest_angle_exp', 'random'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'mnist'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'mpeg7'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'random'),
    os.path.join('output_001_approx', 'delta_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'smallest_angle_exp', 'mnist'),
    os.path.join('output_005_approx', 'smallest_angle_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'smallest_angle_exp', 'random'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'mnist'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'random'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('figs', 'smallest_angle_exp', 'random'),
    os.path.join('figs', 'smallest_angle_exp', 'mnist'),
    os.path.join('figs', 'smallest_angle_exp', 'mpeg7'),
    os.path.join('figs', 'uniform_sample_exp', 'random'),
    os.path.join('figs', 'uniform_sample_exp', 'mnist'),
    os.path.join('figs', 'uniform_sample_exp', 'mpeg7'),
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

def get_map_data(url, target_dir, data_set_name):
    # first check if we should do any downloading
  if os.path.exists(os.path.join(target_dir, data_set_name)):
    print("Warning: directory %s already exists, not re-downloading %s dataset" % (target_dir, data_set_name))
    return

  with tempfile.TemporaryDirectory() as tmp:
    data = wget.download(url,tmp)

    # Pull only maps data
    with ZipFile(data, 'r') as zip_ref:
      file_list = zip_ref.namelist()
      temp_dir = os.path.dirname(zip_ref.namelist()[1])
      for file in file_list:
        if file.startswith('pfoser-mapconstruction-bf67921/data/maps/'):
          zip_ref.extract(file,tmp)
      
      # move and rename
      dst = os.path.join(target_dir, data_set_name)
      os.rename(os.path.join(tmp,temp_dir), dst)

      #Move files up one directory and delete unwanted data folder
      maps_dir = os.path.join(target_dir,data_set_name,'maps')
      shutil.move(os.path.join(target_dir,data_set_name,'data','maps'), maps_dir)
      if os.path.exists(os.path.join(target_dir,data_set_name, 'data')):
        os.rmdir(os.path.join(target_dir,data_set_name, 'data'))
      
      print("Directory " , dst,  " Created ")

  #Unzip files
  unzip_contents(maps_dir)

  return dst

def unzip_contents(dir_name):
  for item in os.listdir(dir_name): 
    if item.endswith('.zip'): 
        file_name = os.path.join(dir_name, item) 
        zip_ref = ZipFile(file_name) 
        zip_ref.extractall(dir_name)
        zip_ref.close()
        os.remove(file_name)
    
def get_mpeg7(url, target_dir):
  dst = get_data(url, target_dir, 'mpeg7')
  
  # clean bad files
  if os.path.exists(os.path.join('data', 'mpeg7', 'rat-09.gif')):
    os.remove('data/mpeg7/rat-09.gif')
  

def get_emnist(url, target_dir):
  dst = get_data(url, target_dir, 'emnist')


def get_maps(url, target_dir):
  dst = get_map_data(url, target_dir, 'map_construction')


def download_data():
  get_mpeg7(URL_MPEG7, dir_data)
  get_emnist(URL_EMNIST, dir_data)
  get_maps(URL_MAP_CONSTRUCTION, dir_data)

def preprocess_data(dir_list):
    make_folders(dir_list)
    download_data()

