import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from kaggle.api.kaggle_api_extended import KaggleApi

#api = KaggleApi()
#api.authenticate()
#datasets = api.dataset_list(search='kaggle-3m')
#for dataset in datasets:
#    print(dataset.ref)
#dataset_ref = 'b33mvssrimanth/kaggle-3m'
#api.dataset_download_files(dataset_ref, path='datasets/kaggle-3m', unzip=True)

BASE_LEN = 89
END_IMG_LEN = 4 
END_MASK_LEN = 9 
IMG_SIZE = 512

DATA_PATH = "./datasets/kaggle-3m/kaggle_3m"

filenames = []
fotos = []
for root, dirs, files in os.walk(DATA_PATH):
    for filename in files:
        full_path = os.path.join(root, filename)
        #print(full_path)
        filenames.append(root)
        fotos.append(full_path)

df = pd.DataFrame({
    "filename": filenames, 
    "foto": fotos
})
print(df.head())
