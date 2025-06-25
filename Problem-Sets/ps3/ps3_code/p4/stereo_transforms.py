import torchvision.transforms as transforms

from problems import StereoRandomFlip


class StereoTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, datum):
        return {
            'left_image': self.transform(datum['left_image']),
            'right_image': self.transform(datum['right_image']),
        }


def get_stereo_transformation(data_augmentation=False, size=(256, 512)):
    if data_augmentation:
        return transforms.Compose([
            StereoTransform(transforms.Resize(size=size)),
            # StereoTransform(transforms.ColorJitter(
            #     brightness=0.25, contrast=0.3, saturation=0.3)),
            StereoRandomFlip(),
            StereoTransform(transforms.ToTensor()),
        ])
    else:
        return transforms.Compose([
            StereoTransform(transforms.Resize(size=size)),
            StereoTransform(transforms.ToTensor()),
        ])
