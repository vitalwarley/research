from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from PIL import Image

class ageFIW(Dataset):
    def __init__(self, data_path, csv_path, transform=None, training=False):
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.csv_path = os.path.join(data_path, csv_path)
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        if self.training:
            df = pd.read_csv(self.csv_path, sep=' ', names=['id', 'anchor', 'positive', 'negative', 'kin', '1'])
        else:
            df = pd.read_csv(self.csv_path, sep=' ', names=['id', 'image_1', 'image_2', 'kin', 'label'])
        return df
    
    def _open_images(self, img):
        images = []
        for age in ['20', '30', '40', '50', '60']:
            image = Image.open(os.path.join(self.data_path, img + age + '.jpg'))
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        return images
    
    def __getitem__(self, index):
        if self.training:
            anchor_img = self._open_images(self.image_paths.iloc[index]['anchor'])
            positive_img = self._open_images(self.image_paths.iloc[index]['positive'])
            negative_img = self._open_images(self.image_paths.iloc[index]['negative'])
        
            return anchor_img, positive_img, negative_img
        else:
            image_1 = self._open_images(self.image_paths.iloc[index]['image_1'])
            image_2 = self._open_images(self.image_paths.iloc[index]['image_2'])
            label = self.image_paths.iloc[index]['label']
            
            return image_1, image_2, label

    def __len__(self):
        return len(self.image_paths)

class ageKinFace(Dataset):
    def __init__(self, data_root, data_folder, fold, transform=None, train=False, kinface_version='I'):
        self.data_path = os.path.join(data_root, f'KinFaceW-{kinface_version}', 'images', data_folder)
        self.fold = fold
        self.train = train
        self.transform = transform
        self.csv_path = os.path.join(data_root, f'KinFaceW-{kinface_version}', f'KinFaceW-{kinface_version}_{data_folder}_age.csv')
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        df = pd.read_csv(self.csv_path) # image1, image2, fold, label
        if self.train:
            df = df[df['fold'] != self.fold]
        else:
            df = df[df['fold'] == self.fold]
        return df
    
    def _open_images(self, img):
        images = []
        for age in ['20', '30', '40', '50', '60']:
            image = Image.open(os.path.join(self.data_path, img + age + '.jpg'))
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        return images
    
    def __getitem__(self, index):
            image_1 = self._open_images(self.image_paths.iloc[index]['image1'])
            image_2 = self._open_images(self.image_paths.iloc[index]['image2'])
            label = self.image_paths.iloc[index]['label']
            
            return image_1, image_2, label

    def __len__(self):
        return len(self.image_paths)
    
class FIW(Dataset):
    def __init__(self, data_path, csv_path, transform=None, training=False):
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.csv_path = os.path.join(data_path, csv_path)
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        if self.training:
            df = pd.read_csv(self.csv_path, sep=' ', names=['id', 'anchor', 'positive', 'negative', 'kin', '1'])
        else:
            df = pd.read_csv(self.csv_path, sep=' ', names=['id', 'image_1', 'image_2', 'kin', 'label'])
        return df
    
    def _open_image(self, img):
        image = Image.open(os.path.join(self.data_path, img))
        if self.transform:
            image = self.transform(image)
        return image
    
    def __getitem__(self, index):
        if self.training:
            anchor_img = self._open_image(self.image_paths.iloc[index]['anchor'])
            positive_img = self._open_image(self.image_paths.iloc[index]['positive'])
            negative_img = self._open_image(self.image_paths.iloc[index]['negative'])
        
            return anchor_img, positive_img, negative_img
        else:
            image_1 = self._open_image(self.image_paths.iloc[index]['image_1'])
            image_2 = self._open_image(self.image_paths.iloc[index]['image_2'])
            label = self.image_paths.iloc[index]['label']
            
            return image_1, image_2, label

    def __len__(self):
        return len(self.image_paths)