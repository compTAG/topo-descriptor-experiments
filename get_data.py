# http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip


import wget
import os
from zipfile import ZipFile

dir_data = 'data'

dir_emnist_005 = 'graphs_005_approx/mnist'
dir_img_emnist_005 = 'graphs_005_approx/mnist_imgs'

dir_emnist_001 = 'graphs_001_approx/mnist'
dir_img_emnist_001 = 'graphs_001_approx/mnist_imgs'

dir_mpeg7_005 = 'graphs_005_approx/mpeg7'
dir_img_mpeg7_005 ='graphs_005_approx/mpeg7_imgs'
dir_mpeg7_extra_005 ='graphs_005_approx/mpeg7_extra'

dir_mpeg7_001 = 'graphs_001_approx/mpeg7'
dir_img_mpeg7_001 ='graphs_001_approx/mpeg7_imgs'
dir_mpeg7_extra_001 ='graphs_001_approx/mpeg7_extra'

dir_list =[dir_data, dir_emnist_005, dir_img_emnist_005, dir_emnist_001, dir_mpeg7_005, 
dir_img_mpeg7_005, dir_mpeg7_extra_005, dir_mpeg7_001,dir_img_mpeg7_001,dir_mpeg7_extra_001]

# Make directories so that generate_graphs runs properly
for x in dir_list:
  if not os.path.exists(x):
    os.makedirs(x)
    print("Directory " , x ,  " Created ")
  else:    
    print("Directory " , x,  " already exists")


# Data url's for mpeg7 and emnist data 
url_mpeg7 = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
url_emnist = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

# Grab the data with wget
mpeg7_data = wget.download(url_mpeg7)
emnist_data = wget.download(url_emnist)

# Unzip mpeg7 data and store in data folder
with ZipFile(mpeg7_data, 'r') as zip_ref:
   zip_ref.extractall('data')

if not os.path.exists('data/mpeg7'):
    os.rename('data/original', 'data/mpeg7')
    print("Directory " , 'data/mpeg7',  " Created ")
    # remove RAT-09 file
    os.remove('data/mpeg7/rat-09.gif')
    os.remove('MPEG7dataset.zip')
else:    
    print("Directory " , 'data/mpeg7' ,  " already exists")

# Unzip emnist data and store in data folder
with ZipFile(emnist_data, 'r') as zip_ref:
   # Extract all the contents of zip file in different directory
   zip_ref.extractall('data')

if not os.path.exists('data/emnist'):
    os.rename('data/matlab', 'data/emnist')
    print("Directory " , 'data/emnist',  " Created ")
    os.remove('matlab.zip')
else:    
    print("Directory " , 'data/emnist' ,  " already exists")  

