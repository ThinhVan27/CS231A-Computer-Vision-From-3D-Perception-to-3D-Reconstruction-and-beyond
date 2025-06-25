import torch
import torch.nn as nn
import torch.nn.functional as F

from problems import bilinear_sampler, generate_image_left, \
        generate_image_right


class Loss(nn.Module):
    def __init__(self, weight_ssim=0.5, weight_smoothness=0.1,
                 weight_lrconsistency=1.0):
        super(Loss, self).__init__()
        self.weight_ssim = weight_ssim
        self.weight_smoothness = weight_smoothness
        self.weight_lrconsistency = weight_lrconsistency

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        _, _, height, width = img.shape
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            scaled_imgs.append(F.interpolate(img,
                               size=[height // ratio, width // ratio],
                               mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def bilinear_sampler_1d_h(self, img, disp):
        return bilinear_sampler(img, disp)

    def _generate_image_left(self, img, disp):
        return generate_image_left(img, disp)

    def _generate_image_right(self, img, disp):
        return generate_image_right(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        def gradient_x(img):
            dx = img[:, :, :, :-1] - img[:, :, :, 1:]
            return dx

        def gradient_y(img):
            gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
            return gy

        disp_gradients_x = [gradient_x(d) for d in disp]
        disp_gradients_y = [gradient_y(d) for d in disp]

        image_gradients_x = [gradient_x(img) for img in pyramid]
        image_gradients_y = [gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradient_x * weight_x
                        for disp_gradient_x, weight_x
                        in zip(disp_gradients_x, weights_x)]
        smoothness_y = [disp_gradient_y * weight_y
                        for disp_gradient_y, weight_y
                        in zip(disp_gradients_y, weights_y)]

        smoothness = [torch.mean(torch.abs(sx)) + torch.mean(torch.abs(sy))
                      for sx, sy in zip(smoothness_x, smoothness_y)]

        return smoothness

    def forward(self, disparities, left_image, right_image):
        # Generate pyramid of images
        left_pyramid = self.scale_pyramid(left_image, len(disparities))
        right_pyramid = self.scale_pyramid(right_image, len(disparities))

        # Store disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in disparities]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disparities]

        # Generate images
        left_est = [self._generate_image_left(image_r, disp_l)
                    for image_r, disp_l in zip(right_pyramid, disp_left_est)]
        right_est = [self._generate_image_right(image_l, disp_r)
                     for image_l, disp_r in zip(left_pyramid, disp_right_est)]

        # LR Consistency
        right_left_disp = [self._generate_image_left(disp_r, disp_l)
                           for disp_r, disp_l
                           in zip(disp_right_est, disp_left_est)]
        left_right_disp = [self._generate_image_right(disp_l, disp_r)
                           for disp_r, disp_l
                           in zip(disp_right_est, disp_left_est)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(lest - lpy))
                   for lest, lpy in zip(left_est, left_pyramid)]
        l1_right = [torch.mean(torch.abs(rest - rpy))
                    for rest, rpy in zip(right_est, right_pyramid)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(lest, lpy))
                     for lest, lpy in zip(left_est, left_pyramid)]
        ssim_right = [torch.mean(self.SSIM(rest, rpy))
                      for rest, rpy in zip(right_est, right_pyramid)]

        image_loss_left = [self.weight_ssim * ssim
                           + (1 - self.weight_ssim) * l1
                           for ssim, l1 in zip(ssim_left, l1_left)]
        image_loss_right = [self.weight_ssim * ssim
                            + (1 - self.weight_ssim) * l1
                            for ssim, l1 in zip(ssim_right, l1_right)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(rl - l)) for rl, l
                        in zip(right_left_disp, disp_left_est)]
        lr_right_loss = [torch.mean(torch.abs(rl - l)) for rl, l
                         in zip(left_right_disp, disp_right_est)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [smoothness / 2 ** i for i, smoothness
                          in enumerate(disp_left_smoothness)]
        disp_right_loss = [smoothness / 2 ** i for i, smoothness
                           in enumerate(disp_right_smoothness)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = (image_loss + self.weight_smoothness * disp_gradient_loss +
                self.weight_lrconsistency * lr_loss)

        return loss
