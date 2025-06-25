import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm

from dataloader import get_dataloader
from model import resnetdisp50
from loss import Loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Data directory.')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model_weight', default=None)
    parser.add_argument('--image_height', type=int, default=256)
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_ssim', type=float, default=0.8)
    parser.add_argument('--weight_smoothness', type=float, default=0.1)
    parser.add_argument('--weight_lrconsistency', type=float, default=1.)
    parser.add_argument('--device', default='cuda:0',
                        help='choose cpu or cuda:0 device"')
    args = parser.parse_args()
    return args


def train(args):
    train_dataloader = get_dataloader(
            args.data_dir, 'train', args.batch_size, args.num_workers,
            data_augmentation=True, size=(args.image_height, args.image_width))
    val_dataloader = get_dataloader(
            args.data_dir, 'val', 1, 0, data_augmentation=False,
            size=(args.image_height, args.image_width))
    device = args.device
    model = resnetdisp50(pretrained=True).to(device)
    if args.model_weight is not None:
        model_weight = torch.load(args.model_weight)
        model.load_state_dict(model_weight)
    loss = Loss(weight_ssim=args.weight_ssim,
                weight_smoothness=args.weight_smoothness,
                weight_lrconsistency=args.weight_lrconsistency).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training loop
        model.train()
        train_losses = []
        for datum in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            left_image = datum['left_image'].to(device)
            right_image = datum['right_image'].to(device)
            disps = model(left_image)
            train_loss = loss(disps, left_image, right_image)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        train_loss_avg = np.mean(train_losses)
        scheduler.step()

        # Validation loop
        model.eval()
        val_losses = []
        for i, datum in enumerate(val_dataloader):
            left_image = datum['left_image'].to(device)
            right_image = datum['right_image'].to(device)
            disps = model(left_image)
            if i == 0:
                plt.imsave(os.path.join(args.output_dir,
                                        f'ep{epoch:03d}_disp.png'),
                           disps[0][0, 0].detach().cpu().numpy(),
                           cmap='plasma')
            val_loss = loss(disps, left_image, right_image)
            val_losses.append(val_loss.item())

        # Save resulting model.
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'model_last.pth'))
        val_loss_avg = np.mean(val_losses)
        print(f'epoch {epoch:03d}:\t'
              f'train loss: {train_loss_avg:.4f}\t'
              f'val loss: {val_loss_avg:.4f}')
        if best_val_loss > val_loss_avg:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'model_best.pth'))
            best_val_loss = val_loss_avg


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
