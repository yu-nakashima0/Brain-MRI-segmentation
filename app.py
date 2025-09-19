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

DATA_PATH = "./datasets/kaggle-3m/kaggle_3m"

filenames = []
fotos = []
imgs = [] 
masks = []
patients = []
for root, dirs, files in os.walk(DATA_PATH):
    for filename in files:
        full_path = os.path.join(root, filename)
        #print(full_path)
        filenames.append(root)
        fotos.append(full_path)
        name_of_patient = root[31:43]
        patients.append(name_of_patient)
        if "mask" in full_path:
            masks.append(full_path)
            imgs.append(0)
        else:
            imgs.append(full_path)
            masks.append(0)

df = pd.DataFrame({
    "Patient": patients, 
    "Filename": filenames, 
    "Photo": fotos,
    "Image": imgs,
    "Mask": masks,
})
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

data = pd.read_csv("./data.csv")

df = pd.merge(df, data, on=["Patient"], how="outer")
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())
print(df.columns)