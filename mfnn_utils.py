from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
from zipfile import ZipFile
import numpy as np

def get_filename (path):
    return path.split('/')[-1]

def retrieve (url, dest_dir='.', desc='', overwrite=False):

    def pbar_updater (block_num, block_size, total_size):
        '''ref: https://github.com/tqdm/tqdm#hooks-and-callbacks'''
        if pbar.total != total_size:
            pbar.total = total_size
        pbar.update(block_num * block_size - pbar.n)
        '''pbar.n is the current total of bytes downloaded'''
        return

    filename = get_filename(url)
    destination = dest_dir + "/" + filename
    if not isfile(destination) or overwrite:
        with tqdm (unit='B', unit_scale=True, miniters=1) as pbar:
            if desc == '': 
                desc = filename
            pbar.set_description(desc)
            urlretrieve(url, destination, pbar_updater)
            pbar.refresh()
    return destination

def unzip(source_filename, dest_dir):
    '''ref: https://docs.python.org/2/library/zipfile.html#zipfile.ZipFile.extract'''
    with ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)        
    return

def pddf_to_nparr(pddf):

    rows = np.array(pddf, ndmin=2)
    arr = [np.array(row, ndmin=2) for row in rows]
    return np.array(arr, ndmin=2)
