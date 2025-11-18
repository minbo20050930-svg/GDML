import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ph2_dataset(Dataset):
    def __init__(self, dataset_folder='/ISIC2018_Task1_npy_all',
                 folder='folder0', train_type='train', with_name=False, transform=None):
        self.transform = transform
        self.train_type = train_type
        self.with_name = with_name
        self.folder_file = './Datasets/' + folder

        if self.train_type in ['train', 'validation', 'test']:
            # this is for cross validation
            with open('Datasets/folder/ph2_test.list',
                      'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_lesion.npy') for x in self.image_list]
            self.mask1 = [join(dataset_folder, 'body-origin', x.split('.')[0] + '.npy') for x in
                          self.image_list]
            self.mask2 = [join(dataset_folder, 'detail-origin', x.split('.')[0] + '.npy') for x in
                          self.image_list]
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.folder) == len(self.mask)

    def __getitem__(self, item: int):
        image = np.load(self.folder[item])
        label = np.load(self.mask[item])
        label1 = np.load(self.mask1[item])
        label2 = np.load(self.mask2[item])

        label1[label1 < 125] = 0
        label1[label1 >= 125] = 255

        label2[label2 < 125] = 0
        label2[label2 >= 125] = 255

        name = self.folder[item].split('/')[-1]

        sample = {'image': image, 'label': label, 'label1': label1, 'label2': label2}

        if self.transform is not None:
            # TODO: transformation to argument datasets randomflip_rotate
            sample = self.transform(sample, self.train_type)

        if self.with_name:
            return name, sample['image'], sample['label'], sample['label1'], sample['label2']
        else:
            return sample['image'], sample['label'], sample['label1'], sample['label2']

    def __len__(self):
        return len(self.folder)