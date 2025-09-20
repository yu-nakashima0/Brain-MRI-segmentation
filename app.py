import os
import cv2
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from kaggle.api.kaggle_api_extended import KaggleApi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

#api = KaggleApi()
#api.authenticate()
#datasets = api.dataset_list(search='kaggle-3m')
#for dataset in datasets:
#    print(dataset.ref)
#dataset_ref = 'b33mvssrimanth/kaggle-3m'
#api.dataset_download_files(dataset_ref, path='datasets/kaggle-3m', unzip=True)

#read dataset
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
        if "mask" in full_path:
            name_of_patient = root[31:43]
            patients.append(name_of_patient)
            masks.append(full_path)
        else:
            imgs.append(full_path)


#make dataframe
df_photo = pd.DataFrame({
    "Patient": patients,
    "Image": imgs,
    "Mask" : masks
})
print(df_photo.isna().sum())

#read csv data 
data = pd.read_csv("./data.csv")
print(data.isna().sum())
print(data.duplicated().sum())


#merge df_photo and data
df = pd.merge(df_photo, data, on=["Patient"], how="outer")
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())
print(df.columns)

#check duplicate and missing value
print(df.isna().sum())
print(df.duplicated().sum())

# Adding A/B column for diagnosis
def positiv_negativ_diagnosis(masks):
    value = np.max(cv2.imread(masks))
    if value > 0 : 
        return 1
    else: 
        return 0

#make new feature "Diagnosis"
df["Diagnosis"] = df["Mask"].apply(lambda m: positiv_negativ_diagnosis(m))
print(df.head())


#Diagnosis plot
sns.countplot(data=df, x = "Diagnosis", palette=["paleturquoise", "salmon"], legend = False)
plt.xticks(ticks=[0,1], labels=["Negativ", "Positiv"])
plt.title("Tumor Detected / Not Detected")
plt.show()


#Correlation plot
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()


#compare the photos of image and mask
image_paths = ["./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_16.tif", "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_16_mask.tif",
               "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_19.tif", "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_19_mask.tif",
               "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_14.tif", "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_14_mask.tif",
               "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4943_20000902\\TCGA_CS_4943_20000902_15.tif", "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4943_20000902\\TCGA_CS_4943_20000902_15_mask.tif",
               "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4943_20000902\\TCGA_CS_4943_20000902_10.tif", "./datasets/kaggle-3m/kaggle_3m\\TCGA_CS_4943_20000902\\TCGA_CS_4943_20000902_10_mask.tif",
               ]
fig, axes = plt.subplots(5, 2, figsize=(10,10))  
axes = axes.flatten()  
for ax, path in zip(axes, image_paths):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    ax.imshow(img)
    ax.axis("off") 

fig.suptitle("Image and Mask", fontsize=16) 
fig.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()


#Preprocessing and Data Augmentation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class BrainMriDataset(Dataset):
    def __init__(self, df, transforms):
        
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df) #total length of dataset
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1]) #get path for image of certain index(idx) and read image as BGR format
        mask = cv2.imread(self.df.iloc[idx, 2], 0) #get path for mask of certain index(idx) and read mask as BGR format

        augmented = self.transforms(image=image, mask=mask) #Apply the same random transformation to both the image and the mask
 
        image = augmented['image']
        mask = augmented['mask']   
        
        return image, mask # return image and mask which were processed


#transform
PATCH_SIZE = 128 #final size of photo : 128 * 128

#make a pipeliene for training -> diversify training data ->  prevent overfitting
strong_transforms = A.Compose([
    A.RandomResizedCrop(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    
    # Pixels
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.25),
    A.IAAEmboss(p=0.25),
    A.Blur(p=0.01, blur_limit = 3),
    
    # Affine
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
    ], p=0.8),
    
    
    A.Normalize(p=1.0),
  
    ToTensorV2(),
])

# make a pipeline for test / validation
transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

    A.Normalize(p=1.0),
    ToTensorV2(),
])
    
