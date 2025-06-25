import time  
import argparse as arg 
import datetime
import os

import torch  
import torch.nn as nn  
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils  
from tensorboardX import SummaryWriter

import data
from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion
from utils import AverageMeter, colorize, init_training_vals

def train(epochs, 
        train_data_loader,
        test_data_loader,
        lr=0.0001, 
        save="checkpoints/", 
        theta=0.1, 
        device="cuda", 
        pretrained=False,
        checkpoint=None,
        model=None,
        start_epoch=0):

    num_trainloader = len(train_data_loader)
    num_testloader = len(test_data_loader)

    # Training utils  
    model_prefix = "monocular_"
    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    theta = theta
    save_count = 0
    epoch_loss = []
    batch_loss = []
    sum_loss = 0

    if model is not None:
        print("Using passed in model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        if checkpoint:
          print("Loading from checkpoint ...")
        else:
          print("Initializing fresh model ...")
        
        model, optimizer, start_epoch = init_training_vals(pretrained=pretrained,
                                                            epochs=epochs,
                                                            lr=lr,
                                                            ckpt=checkpoint, 
                                                            device=device                                            
                                                            )
        print("Resuming from: epoch #{}".format(start_epoch))

        model = model.to(device)

    
    log_dir = 'runs'
    # Logging
    writer = SummaryWriter(log_dir,comment="{}-training".format(model_prefix))
    
    # Loss functions 
    l1_criterion = nn.L1Loss() #TODO initialize the L1 Loss to be used for optimizing the output depth

    # Starting training 
    print("Starting training ... ")
     
    for epoch in range(start_epoch, epochs):
        
        batch_time = AverageMeter() 
        l1_loss_meter = AverageMeter() 
        gradient_loss_meter = AverageMeter() 
        ssim_loss_meter = AverageMeter() 
        net_loss_meter = AverageMeter() 
        model.train()
        epoch_start = time.time()
        end = time.time()

        for idx, batch in enumerate(train_data_loader):

            optimizer.zero_grad() 

            image_x = batch["rgb"].to(device)
            depth_y = batch["depth"].to(device)

            preds = model(image_x) # TODO call your model on the image input to get its predictions

            # calculating the losses 
            l1_loss = l1_criterion(preds, depth_y) # TODO call the l1_criterion with the predictions and normalized depth
            
            ssim_loss = torch.clamp(
                (1-ssim_criterion(preds, depth_y, 1.0))*0.5, 
                min=0, 
                max=1
            )

            # Gradient loss checks for the edges of the image to be similar
            gradient_loss = gradient_criterion(depth_y, preds, device=device)

            net_loss = ((1.0 * ssim_loss) + (torch.mean(gradient_loss)) + 
                       (theta * torch.mean(l1_loss)))
            
            batch_size = image_x.size(0)
            l1_loss_meter.update(theta * torch.mean(l1_loss).data.item(), batch_size)
            gradient_loss_meter.update(torch.mean(gradient_loss).data.item(), batch_size)
            ssim_loss_meter.update(ssim_loss.data.item(), batch_size)
            net_loss_meter.update(float(net_loss.data.item()), batch_size)

            net_loss.backward()
            optimizer.step()

            # Time metrics 
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(num_trainloader-idx))))

            # Logging  
            num_iters = epoch * num_trainloader + idx + 1 
            if (idx + 1 )% 5 == 0 :
                print(
                    "Epoch: #{0} Batch: {1}/{2}\t"
                    "Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t"
                    "eta {eta}\t"
                    "LOSS (current/average) {loss.val:.4f}/{loss.avg:.4f}\t"
                    .format(epoch+1, idx+1, 
                            num_trainloader, 
                            batch_time=batch_time, 
                            eta=eta, 
                            loss=net_loss_meter)
                )

                writer.add_scalar("Train/L1 loss", l1_loss_meter.val, num_iters)
                writer.add_scalar("Train/Gradient Loss", gradient_loss_meter.val, num_iters)
                writer.add_scalar("Train/SSIM Loss", ssim_loss_meter.val, num_iters)
                writer.add_scalar("Train/Net Loss", net_loss_meter.val, num_iters)
            
            if (idx + 1) % 20 == 0: 
                log_progress_images(model, writer, test_data_loader, num_iters, device)

            if (idx + 1) % 100 == 0:
                ckpt_path = save+"ckpt_{}.pth".format(epoch)
                torch.save({
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict":  optimizer.state_dict(),
                }, ckpt_path) 
            del image_x
            del depth_y
            del preds   
        
        test_net_loss_meter = AverageMeter() 
        model.eval()
        for idx, batch in enumerate(test_data_loader):

            image_x = batch["rgb"].to(device)
            depth_y = batch["depth"].to(device)

            with torch.no_grad():
                preds = model(image_x)

            # calculating the losses 
            l1_loss = l1_criterion(preds, depth_y)
            ssim_loss = torch.clamp(
                (1-ssim_criterion(preds, depth_y, 1.0))*0.5, 
                min=0, 
                max=1
            )
            gradient_loss = gradient_criterion(depth_y, preds, device=device)

            test_net_loss = ((1.0 * ssim_loss) + (torch.mean(gradient_loss)) + 
                       (theta * torch.mean(l1_loss)))
            
            batch_size = image_x.size(0)
            test_net_loss_meter.update(test_net_loss.data.item(), batch_size)

            del image_x
            del depth_y
        writer.add_scalar("Test/Net Loss", test_net_loss_meter.avg, num_iters)

        print(
            "----------------------------------\n"
            "Epoch: #{0}, Avg. Net Test Loss: {test_avg_loss:.4f}\n" 
            "----------------------------------"
            .format(
                epoch+1, test_avg_loss=test_net_loss_meter.avg
            )
        )

def log_progress_images(model, writer, test_loader, num_iters, device):
    
    """ To record intermediate results of training""" 

    model.eval() 
    sequential = test_loader
    sample_batched = next(iter(sequential))
    
    image = torch.Tensor(sample_batched["rgb"]).to(device)
    depth = torch.Tensor(sample_batched["depth"]).to(device)
    
    inv_normalize_color, inv_normalize_depth = data.get_inverse_transforms()
    if num_iters == 0:
        writer.add_image("Train.1.Image", vision_utils.make_grid(inv_normalize_color(image).data, 
                                                                 nrow=4, normalize=True), num_iters)
    if num_iters == 0:
        writer.add_image("Train.2.Image", colorize(vision_utils.make_grid(inv_normalize_depth(depth).data, 
                                                                 nrow=4, normalize=False)), num_iters)
    
    with torch.no_grad():
        output = model(image)

    writer.add_image("Train.3.Ours", colorize(vision_utils.make_grid(inv_normalize_depth(output).data, 
                                                                    nrow=4, normalize=False)), num_iters)
    writer.add_image("Train.4.Diff", colorize(vision_utils.make_grid(torch.abs(output-depth).data, 
                                                                    nrow=4, normalize=False), cmap='magma'), num_iters)
    
    del image
    del depth
    del output
