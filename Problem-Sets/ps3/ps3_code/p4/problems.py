""" Please fill in your codes for problem set in this file. """
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvutils
import torch.nn.functional as F


"""
Problem a. Implement horizontal flip data augmentation.

In neural networks, data augmentation takes a crucial role in better
generalization of the problem. One of the most common data augmentation when
using 2D images as input is to randomly flip the image horizontally. One
interesting difference in our problem setup is that we take a pair of rectified
stereo images as input. In order to maintain the stereo relationship after the
horizontal flip, it requires a special attention.
Fill in the code below, which is a data augmentation class in PyTorch which
takes datum=dict{'left_image': img_l, 'right_image': img_r} as input and
outputs the same dictionary with random flip at 50% of chance.  Use
`self.transform` fuction to horizontally flip a single image.
"""


class StereoRandomFlip:
    def __init__(self):
        # Flips image horizontally (not random).
        self.transform = transforms.RandomHorizontalFlip(p=1)

    def _flip(self, left_image, right_image):
        # Use `self.transform` to flip the image horizontally.

        raise NotImplementedError

        return flipped_left_image, flipped_right_image

    def __call__(self, datum):
        if np.random.rand() < 0.5:
            flipped_left_image, flipped_right_image = self._flip(
                    datum['left_image'], datum['right_image'])

            return {
                'left_image': flipped_left_image,
                'right_image': flipped_right_image,
            }
        return datum


"""
Problem b. Implement bilinear sampler.

Implement a function bilinear_sampler which shifts the given horizontally given
the disparity. The core idea of unsupervised monocular depth estimation is that
we can generate left image from right and vice versa by sampling rectified images
horizontally using the disparity. We will ask you to implement a function that
simply samples image with horizontal displacement as given by the input
disparity.

The input to this function is as following:

Inputs:
    img: torch.Tensor (batch_size, n, height, width).
        Input image to transform from. n can be 3 for RGB images and 1 for
        disparity images.
    disp: torch.Tensor (batch_size, 1, height, width).
        Input disparity map of values -1 < disp < 1
Output:
    torch.Tensor (batch_size, n, height, width).
    The sampled image.

Since the input images are rectified, the input data doesn't shift vertically.
We only need to compute how the image transforms horizontally given the
disparity map. We define the coordinate system of our image as 0-1. Then, we
sample image with horizontal displacement as given by the input disparity
value.  For example, if the entire disparity map has value of 0.5, the entire
image will be shifted toward left by half. If the entire disparity map has
value of -0.5, the entire image will be shifted toward right by half. Note that
the disparity value can be negative, which shifts the image toward left.

More technically,
1. Generate 0-1 xy coordinates using `torch.meshgrid`
2. Add disparity to the x coordinate grid.
3. Sample image from disparity-applied grid using `F.grid_sample`. Note that
   `F.grid_sample` expects grid of range -1 to 1 and we need to scale our grid
   accordingly.
"""


def bilinear_sampler(img, disp):
    raise NotImplementedError
    return output


"""
Problem c. Implement left/right image generator.

Next, simply apply the bilinear sampler to generate image from left to right
and vice versa. Note that the network disparity prediction `disp` is always
positive and you will will to flip signs accordingly depending on the direction
you want to shift the image to.

Inputs:
    img: torch.Tensor (batch_size, n, height, width).
        Input image to transform from. n can be 3 for RGB images and 1 for
        disparity images.
    disp: torch.Tensor (batch_size, 1, height, width).
        Input disparity map of values 0 < disp < 1
Output:
    torch.Tensor (batch_size, n, height, width).
    The sampled image.
"""


def generate_image_right(img, disp):
    raise NotImplementedError
    return output


def generate_image_left(img, disp):
    raise NotImplementedError
    return output


if __name__ == '__main__':
    data = np.load('data.npz')
    disparities = data['disparities']
    left_images = data['left_image']
    right_images = data['right_image']
    left_image_t = torch.from_numpy(left_images[0])
    right_image_t = torch.from_numpy(right_images[0])

    # Problem a. Implement horizontal flip data augmentation.
    transform_flip = StereoRandomFlip()
    flipped_left_t, flipped_right_t = transform_flip._flip(
            left_image_t, right_image_t)

    tvutils.save_image(left_image_t, 'a_input_left.png')
    tvutils.save_image(right_image_t, 'a_input_right.png')
    tvutils.save_image(flipped_left_t, 'a_flipped_left.png')
    tvutils.save_image(flipped_right_t, 'a_flipped_right.png')

    # Problem b. Implement bilinear sampler.
    img_left = torch.from_numpy(left_images)
    img_right = torch.from_numpy(right_images)

    shift_left = torch.ones_like(img_left) * 0.5
    shift_right = torch.ones_like(img_left) * -0.5

    img_shift_left_half = bilinear_sampler(img_left, shift_left)
    img_shift_right_half = bilinear_sampler(img_left, shift_right)
    tvutils.save_image(img_left[0], 'b_input_img.png')
    tvutils.save_image(img_shift_left_half[0], 'b_shift_left_half.png')
    tvutils.save_image(img_shift_right_half[0], 'b_shift_right_half.png')

    # Problem b. left/right image generator.
    disp_l = torch.from_numpy(disparities[0, 0]).unsqueeze(0).unsqueeze(0)
    disp_r = torch.from_numpy(disparities[0, 1]).unsqueeze(0).unsqueeze(0)
    img_left_est = generate_image_left(img_right, disp_l)
    img_right_est = generate_image_right(img_left, disp_r)
    disp_left_est = generate_image_left(disp_r, disp_l)
    disp_right_est = generate_image_right(disp_l, disp_r)

    tvutils.save_image(left_image_t, 'c_input_left.png')
    tvutils.save_image(right_image_t, 'c_input_right.png')
    tvutils.save_image(img_left_est[0], 'c_shift_image_left.png')
    tvutils.save_image(img_right_est[0], 'c_shift_image_right.png')
    plt.imsave('c_input_disp_left.png', disparities[0, 0], cmap='plasma')
    plt.imsave('c_input_disp_right.png', disparities[0, 1], cmap='plasma')
    plt.imsave('c_shift_disp_right.png', disp_right_est[0, 0], cmap='plasma')
    plt.imsave('c_shift_disp_left.png', disp_left_est[0, 0], cmap='plasma')
