import wget
import os
import tempfile
from zipfile import ZipFile
import shutil
import requests
import certifi

URL_MPEG7 = 'https://www.ehu.eus/ccwintco/uploads/d/de/MPEG7_CE-Shape-1_Part_B.zip'
URL_EMNIST = 'https://rds.westernsydney.edu.au/Institutes/MARCS/BENS/EMNIST/emnist-matlab.zip'
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
    os.path.join('output_001_approx', 'smallest_stratum_exp', 'mnist'),
    os.path.join('output_001_approx', 'smallest_stratum_exp', 'mpeg7'),
    os.path.join('output_001_approx', 'smallest_stratum_exp', 'random'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'mnist'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'mpeg7'),
    os.path.join('output_001_approx', 'uniform_sample_exp', 'random'),
    os.path.join('output_001_approx', 'delta_exp', 'mpeg7'),
    os.path.join('output_001_approx', 'delta_exp', 'mnist'),   
    os.path.join('output_005_approx', 'delta_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'delta_exp', 'mnist'),
    os.path.join('figs','delta_exp_figs','mnist'),
    os.path.join('figs','delta_exp_figs','mpeg7'),
    os.path.join('output_005_approx', 'smallest_stratum_exp', 'mnist'),
    os.path.join('output_005_approx', 'smallest_stratum_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'smallest_stratum_exp', 'random'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'mnist'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'mpeg7'),
    os.path.join('output_005_approx', 'uniform_sample_exp', 'random'),
    os.path.join('analysis_005_approx', 'smallest_stratum_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'smallest_stratum_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'smallest_stratum_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'smallest_stratum_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'smallest_stratum_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'smallest_stratum_exp', 'combined_data', 'random'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('figs', 'smallest_stratum_exp', 'random'),
    os.path.join('figs', 'smallest_stratum_exp', 'mnist'),
    os.path.join('figs', 'smallest_stratum_exp', 'mpeg7'),
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
    # Check if the target directory already exists
    if os.path.exists(os.path.join(target_dir, data_set_name)):
        print("Warning: directory %s already exists, not re-downloading %s dataset" % (target_dir, data_set_name))
        return

    # Download the data using requests
    response = requests.get(url, verify=False)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        # Write the downloaded content to a temporary file
        temp_file_path = os.path.join(tmp, "emnist-matlab.zip")
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        
        # Unzip data and store in data folder
        with ZipFile(temp_file_path, 'r') as zip_ref:
            # Extract data to temporary directory
            zip_ref.extractall(tmp)
            temp_dir = os.path.dirname(zip_ref.namelist()[1])
            
            # Move and rename extracted directory
            dst = os.path.join(target_dir, data_set_name)
            os.rename(os.path.join(tmp, temp_dir), dst)
            print("Directory ", dst, " Created ")
    
    return dst
    
def get_mpeg7(url, target_dir):
  dst = get_data(url, target_dir, 'mpeg7')
  
  # clean bad files
  if os.path.exists(os.path.join('data', 'mpeg7', 'rat-09.gif')):
    os.remove('data/mpeg7/rat-09.gif')
  

def get_emnist(url, target_dir):
  dst = get_data(url, target_dir, 'emnist')


def make_delta_headers():
  with open(os.path.join("output_001_approx","delta_exp", "mnist","deltas.txt"), 'w') as f:
    f.write("n,delta,outFile\n")
    f.close()
  with open(os.path.join("output_001_approx","delta_exp", "mpeg7","deltas.txt"), 'w') as f:
    f.write("n,delta,outFile\n")
    f.close()
  with open(os.path.join("output_005_approx","delta_exp", "mnist","deltas.txt"), 'w') as f:
    f.write("n,delta,outFile\n")
    f.close()
  with open(os.path.join("output_005_approx","delta_exp", "mpeg7","deltas.txt"), 'w') as f:
    f.write("n,delta,outFile\n")
    f.close()
  
def download_data():
  get_mpeg7(URL_MPEG7, dir_data)
  get_emnist(URL_EMNIST, dir_data)

def preprocess_data(dir_list):
    make_folders(dir_list)
    download_data()
    make_delta_headers()

preprocess_data(dir_list)
