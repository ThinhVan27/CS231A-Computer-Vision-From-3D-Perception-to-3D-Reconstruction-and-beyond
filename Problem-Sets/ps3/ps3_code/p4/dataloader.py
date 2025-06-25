import os
import glob

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from stereo_transforms import get_stereo_transformation


class KittiStereoDataset(Dataset):
    def __init__(self, data_dir, subset, transforms=None):
        self.left_images = sorted(glob.glob(os.path.join(
            data_dir, subset, '*', 'image_02', 'data', '*.png')))
        self.right_images = sorted(glob.glob(os.path.join(
            data_dir, subset, '*', 'image_03', 'data', '*.png')))
        assert len(self.left_images) == len(self.right_images)
        self.transforms = transforms

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, i):
        left_image = Image.open(self.left_images[i])
        right_image = Image.open(self.right_images[i])
        datum = {'left_image': left_image, 'right_image': right_image}
        if self.transforms is not None:
            datum = self.transforms(datum)
        return datum


def get_dataloader(data_dir, subset, batch_size, num_workers,
                   data_augmentation=False, size=(256, 512)):
    transformation = get_stereo_transformation(
            data_augmentation=data_augmentation, size=size)
    dataset = KittiStereoDataset(data_dir, subset,
                                 transforms=transformation)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=data_augmentation, num_workers=num_workers)
    return dataloader
