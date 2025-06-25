import matplotlib.pyplot as plt
import random
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MNISTDatasetWrapper(Dataset):
    """
    A Dataset for learning with subsets of the Fashion MNIST dataset for either the
    original labels or labels that describe how the image has been rotated.
    Rotations will be applied clockwise, with a random choice of one of the
    following degrees: [0, 45, 90, 135, 180, 225, 270, 315]

    - original_dataset - the fashion mnist dataset we got with torchvision
    - pct - percent of data to use
    - for_rotation_classification - True=Use rotation labels.
                                    False=Use original classification labels.
    """

    def __init__(self, original_dataset, pct=1.0, for_rotation_classification=False):
        self.dataset = original_dataset
        self.dataset_size = int(len(self.dataset)*pct)
        self.for_rotation_classification = for_rotation_classification
        self.label_to_class = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.imageToTensor = transforms.ToTensor()
        self.tensorToImage = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.2859,), (0.3530,))
        self.denormalize = transforms.Normalize((-0.2859/0.3530,), (1.0/0.3530,))
        self.rot_choices = [0, 45, 90, 135, 180, 225, 270, 315]

    def __len__(self):
        # TODO: Implement this function.
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Returns a 32x32 MNIST digit and its corresponding clothes
        label, or if self.for_rotation_classification is true returns
        the image rotated by a random rotation amount from
        self.rot_choices. Note: the label should be a number between 0-7,
        not the number of degrees to rotate by.
        """
        img, label = self.dataset[idx]
        if not self.for_rotation_classification:
            return self.normalize(img), torch.tensor(label).long()
        else:
            img = self.tensorToImage(img)
            label = random.randint(0, len(self.rot_choices)-1) # TODO choose a label for rotation
            img = img.rotate(self.rot_choices[label]) # TODO use PIL's rotate function to modify the image according to the label 
            img = self.normalize(self.imageToTensor(img))
            return img, torch.tensor(label).long()

    def show_batch(self, n=3):
        fig, axs = plt.subplots(n, n)
        fig.tight_layout()
        for i in range(n):
            for j in range(n):
                rand_idx = random.randint(0, len(self)-1)
                img, label = self.__getitem__(rand_idx)
                axs[i, j].imshow(self.tensorToImage(self.denormalize(img)), cmap='gray')
                if not self.for_rotation_classification:
                    axs[i, j].set_title('Label: {0} (#{1})'.format(label.item(), self.label_to_class[label.item()]))
                else:
                    axs[i, j].set_title('Label: {0} ({1} Degrees)'.format(label.item(), label.item()*45))
                axs[i, j].axis('off')
