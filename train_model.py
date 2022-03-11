import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from dataset import *
#from dataset import make_dataloaders
from loss import *
from  models import *
from utils import *
from pre_train import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dl, val_dl, epochs, display_every=501):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in train_dl:
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                #visualize(model, data, save=False) # function displaying the model's outputs

        # After finish training for this epoch, compute train & val accuracy
        print('--- Stats for epoch {} ---'. format(e))            
        log_results(loss_meter_dict)
        print('Computing TRAINING accuracies...')
        train_psnr, train_ssim = compute_acc(model.net_G, train_dl)
        print('Train PSNR: {:.4f}'.format(train_psnr))
        print('Train SSIM: {:.4f}'.format(train_ssim))
        print('Computing VALIDATION accuracies...')
        val_psnr, val_ssim = compute_acc(model.net_G, val_dl)
        print('Val PSNR: {:.4f}'.format(val_psnr))
        print('Val SSIM: {:.4f}'.format(val_ssim))

        save_statistics(loss_meter_dict, train_psnr, train_ssim, val_psnr, val_ssim, e)

        if e in [0, 24, 49]:
            PATH_G = "./baseline_netG_epoch{}.pth".format(e)
            torch.save(model.net_G.state_dict(), PATH_G)
            PATH_D = "./baseline_netD_epoch{}.pth".format(e)
            torch.save(model.net_D.state_dict(), PATH_D)

def run_experiment():
    train_dl = make_dataloaders(batch_size=8, n_workers=1,paths=train_paths, split='train')
    val_dl = make_dataloaders(batch_size=8, n_workers=1, paths=val_paths, split='val')
    print('Dataloaders created...')

    # device config
    device = torch.cuda.current_device()
    print('Using GPU', device)

    model = MainModel()
    num_of_epochs = 50
    print('Training model for {} epochs...'.format(num_of_epochs))
    train_model(model, train_dl, val_dl, num_of_epochs) # Run 50 epochs    

    # net_G = build_res_unet(resnet, n_input=1, n_output=2, size=256)
    # opt = optim.Adam(net_G.parameters(), lr=learningRate,weight_decay=weight_decay)
    # criterion = nn.L1Loss()
    # pretrain_generator(net_G, train_dl,val_dl, opt, criterion, 20, resnet, learningRate,weight_decay)

#net_G = build_res_unet(n_input=1, n_output=2, size=256)
#opt = optim.Adam(net_G.parameters(), lr=1e-4)
#criterion = nn.L1Loss()        
#pretrain_generator(net_G, train_dl,val_dl, opt, criterion,40)
#torch.save(net_G.state_dict(), "res18-unet.pt")


#net_G = build_res_unet(n_input=1, n_output=2, size=256)
#net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
#model = MainModel(net_G=net_G)
#train_model(model, train_dl, 1)

