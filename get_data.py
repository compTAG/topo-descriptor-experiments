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

#Make data directory
if not os.path.exists(dir_data):
    os.makedirs(dir_data)
    print("Directory " , dir_data ,  " Created ")
else:    
    print("Directory " , dir_data,  " already exists")    

# Make emnist 005 directories

if not os.path.exists(dir_emnist_005):
    os.makedirs(dir_emnist_005)
    print("Directory " , dir_emnist_005 ,  " Created ")
else:    
    print("Directory " , dir_emnist_005 ,  " already exists")  

if not os.path.exists(dir_img_emnist_005):
    os.makedirs(dir_img_emnist_005)
    print("Directory " , dir_img_emnist_005 ,  " Created ")
else:    
    print("Directory " , dir_img_emnist_005 ,  " already exists") 

# Make emnist 001 directories

if not os.path.exists(dir_emnist_001):
    os.makedirs(dir_emnist_001)
    print("Directory " , dir_emnist_001 ,  " Created ")
else:    
    print("Directory " , dir_emnist_001 ,  " already exists") 

if not os.path.exists(dir_img_emnist_001):
    os.makedirs(dir_img_emnist_001)
    print("Directory " , dir_img_emnist_001 ,  " Created ")
else:    
    print("Directory " , dir_img_emnist_001 ,  " already exists") 


# Make mpeg7 005 directories

if not os.path.exists(dir_mpeg7_005):
    os.makedirs(dir_mpeg7_005)
    print("Directory " , dir_mpeg7_005 ,  " Created ")
else:    
    print("Directory " , dir_mpeg7_005 ,  " already exists")


if not os.path.exists(dir_img_mpeg7_005):
    os.makedirs(dir_img_mpeg7_005)
    print("Directory " , dir_img_mpeg7_005 ,  " Created ")
else:    
    print("Directory " , dir_img_mpeg7_005 ,  " already exists")

if not os.path.exists(dir_mpeg7_extra_005):
    os.makedirs(dir_mpeg7_extra_005)
    print("Directory " , dir_mpeg7_extra_005 ,  " Created ")
else:    
    print("Directory " , dir_mpeg7_extra_005 ,  " already exists")

# Make mpeg7 001 directories

if not os.path.exists(dir_mpeg7_001):
    os.makedirs(dir_mpeg7_001)
    print("Directory " , dir_mpeg7_001 ,  " Created ")
else:    
    print("Directory " , dir_mpeg7_001 ,  " already exists")

if not os.path.exists(dir_img_mpeg7_001):
    os.makedirs(dir_img_mpeg7_001)
    print("Directory " , dir_img_mpeg7_001 ,  " Created ")
else:    
    print("Directory " , dir_img_mpeg7_001 ,  " already exists")

if not os.path.exists(dir_mpeg7_extra_001):
    os.makedirs(dir_mpeg7_extra_001)
    print("Directory " , dir_mpeg7_extra_001 ,  " Created ")
else:    
    print("Directory " , dir_mpeg7_extra_001 ,  " already exists")


# Data url's
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

