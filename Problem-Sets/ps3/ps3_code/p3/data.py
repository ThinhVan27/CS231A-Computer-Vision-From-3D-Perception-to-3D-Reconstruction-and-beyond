from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np  
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations

def load_zip_to_mem(zip_file, is_mono=True):
    """
    Function to load CLEVR-D data from the zip file.
    """
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    file_dict = {name.split('/')[1]: input_zip.read(name) for 
            name in input_zip.namelist() if '.png' in name}
    data = []
    for file_name in file_dict:
      #Only deal with right rgb images, all else via dict lookup
      if 'right' in file_name and 'CLEVR-D' not in file_name:
        rgb_right = file_dict[file_name]
        right_depth_name = file_name.replace('CLEVR','CLEVR-D')
        depth_right = file_dict[right_depth_name]
        if is_mono:
          data.append( (rgb_right, depth_right))
        else:
          rgb_left = file_dict[file_name.replace('right','left')]
          depth_left = file_dict[right_depth_name.replace('right','left')]
          data.append( (rgb_right,rgb_left, depth_right,depth_left))
    return data

def get_inverse_transforms():
    """
    Get inverse transforms to undo data normalization
    """
    inv_normalize_color = transforms.Normalize(
    mean=[-0.462/0.094, -0.467/0.096, -0.469/0.101],
    std=[1/0.094, 1/0.096, 1/0.101]
    )
    inv_normalize_depth = transforms.Normalize(
    mean=[-0.480/0.295],
    std=[1/0.295]
    )

    return inv_normalize_color, inv_normalize_depth

def get_tensor_to_image_transforms():
    """
    Get transforms to go from Pytorch Tensors to PIL images that can be displayed
    """
    tensor_to_image = transforms.ToPILImage()
    inv_normalize_color, inv_normalize_depth = get_inverse_transforms()
    return (transforms.Compose([inv_normalize_color,tensor_to_image]),
            transforms.Compose([inv_normalize_depth,tensor_to_image]))

class DepthDatasetMemory(Dataset):
    """
    The Dataset class 

    Arguments:
        data (int): list of tuples with data from the zip files
        is_mono (boolen): whether to return monocular or stereo data
        start_idx (int): start of index to use in data list  
        end_idx (int): end of index to use in data list
    """
    def __init__(self, data, is_mono=True, start_idx=0, end_idx = None):
        self.is_mono = is_mono
        self.start_idx = 0
        if end_idx is None:
            end_idx = len(data)
        self.end_idx = end_idx
        self.data = list(data[start_idx:end_idx])

        self.color_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.462, 0.467, 0.469), (0.094, 0.096, 0.101)),
        ])

        self.depth_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.480), (0.295)),
        ])

        self.samples = []
        for idx in range(len(self.data)):
            sample = self.data[idx]
            rgb_right = self.color_transform(Image.open(BytesIO(sample[0])).convert('RGB'))
            depth_right = self.depth_transform(Image.open(BytesIO(sample[1])).convert('L'))
            if self.is_mono:
              sample = {'rgb': rgb_right, 'depth': depth_right}
            else:
              rgb_left = self.color_transform(Image.open(BytesIO(sample[2])).convert('RGB'))
              depth_left = self.depth_transform(Image.open(BytesIO(sample[3])).convert('L'))

              sample = {'rgb_right': rgb_right, 'depth_right': depth_right,
                        'rgb_left': rgb_left, 'depth_left': depth_left}
            self.samples.append(sample)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

def get_data_loaders(path, 
                    is_mono=True, 
                    batch_size=16, 
                    train_test_split=0.8, 
                    pct_dataset=1.0):
    """
    The function to return the Pytorch Dataloader class to iterate through
    the dataset. 

    Arguments:
        is_mono (boolen): whether to return monocular or stereo data
        batch_size (int): batch size for both training and testing 
        train_test_split (float): ratio of data from training to testing
        pct_dataset (float): percent of dataset to use 
    """
    data = load_zip_to_mem(path)
    train_start_idx = 0 # TODO
    train_end_idx = int(train_test_split * len(data) * pct_dataset)# TODO 
    test_start_idx = train_end_idx + 1# TODO 
    test_end_idx = int(len(data) * pct_dataset) # TODO 

    training_dataset = DepthDatasetMemory(data=data, is_mono=is_mono, start_idx=train_start_idx, end_idx=train_end_idx) # TODO 
    testing_dataset = DepthDatasetMemory(data=data, is_mono=is_mono, start_idx=test_start_idx, end_idx=test_end_idx) # TODO 

    return (DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True),
            DataLoader(testing_dataset, batch_size, shuffle=False, pin_memory=True))
