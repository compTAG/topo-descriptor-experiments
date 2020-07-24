import wget
import os
from backports import tempfile
from zipfile import ZipFile

import path

URL_MPEG7 = 'http://www.dabi.temple.edu/~shape/MPEG7/MPEG7dataset.zip'
URL_EMNIST = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'


def download(url, target_dir):
    # first check if we should do any downloading
    if os.path.exists(target_dir):
        print("Warning: directory \"%s\" already exists, not re-downloading dataset" % target_dir)
        return


    with tempfile.TemporaryDirectory() as tmp:
        data = wget.download(url,tmp)

        # Unzip data and store in data folder
        with ZipFile(data, 'r') as zip_ref:
            zip_ref.extractall(tmp)
            temp_dir = os.path.dirname(zip_ref.namelist()[1])

        # move and rename
        os.rename(os.path.join(tmp,temp_dir), target_dir)
        print("Directory %s created" % target_dir)


def get_mpeg7(url, target_dir):
    dst = download(url, target_dir)

    # clean bad files
    rat9 = os.path.join(target_dir, 'rat-09.gif')
    if os.path.exists(rat9):
        os.remove(rat9)


def get_emnist(url, target_dir):
    download(url, target_dir)


def fetch(path_manager):
    path.FolderMaker().make_folder(path_manager.data_dir)

    get_mpeg7(URL_MPEG7, path_manager.mpeg7_data_dir)
    get_emnist(URL_EMNIST, path_manager.mnist_data_dir)

