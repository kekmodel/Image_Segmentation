import csv
import os
import time
import datetime

import numpy as np
import torch
import torchvision
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from evaluation import *
from network_test import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from misc import get_cosine_schedule_with_warmup

import segmentation_models_pytorch as smp


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        wandb.init(name=config.name, project="segmentation", config=config)
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        # self.beta1 = config.beta1
        # self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        # self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.num_total_steps = int(len(train_loader) * config.num_epochs)
        self.warmup_steps = int(self.num_total_steps * config.warmup_rate)

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)
        else:
            self.unet = smp.Unet(encoder_name="timm-efficientnet-b8",
                                 encoder_weights="advprop",
                                 decoder_use_batchnorm=True,  
                                 in_channels=3,                  
                                 classes=1)

        # self.optimizer = optim.Adam(list(self.unet.parameters()),
        #                             self.lr, [self.beta1, self.beta2])
        # print([n for n, p in self.unet.named_parameters()])
        # no_decay = ['bn', 'bias']
        # grouped_parameters = [
        #     {'params': [p for n, p in self.unet.named_parameters() if not any(
        #         nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        #     {'params': [p for n, p in self.unet.named_parameters() if any(
        #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        self.optimizer = optim.Adam(self.unet.parameters(), self.lr)
        # self.scheduler = get_cosine_schedule_with_warmup(
        #     self.optimizer, self.warmup_steps, self.num_total_steps)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=20, verbose=True)
        self.unet.to(self.device)
        wandb.watch(self.unet)
        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img*255
        return img
    
    def dice_loss(self, pred, target):
        eps = 1e-6
        dice = (2. * (pred * target).sum(dim=-1, keepdim=True) + eps) / ((pred + target).sum(dim=-1, keepdim=True) + eps)
        return (1. - dice).mean()

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f.pkl' % (self.model_type,
                                                                              self.num_epochs, 
                                                                              self.lr, 
                                                                              self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.

            for epoch in range(self.num_epochs):

                # self.unet.eval()
                self.unet.train()
                epoch_loss = 0
                # acc = 0.  # Accuracy
                # SE = 0.		# Sensitivity (Recall)
                # SP = 0.		# Specificity
                # PC = 0. 	# Precision
                # F1 = 0.		# F1 Score
                # JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                # length = 0

                for i, (images, GT) in tqdm(enumerate(self.train_loader)):
                    # GT : Ground Truth

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    # print(SR_flat.shape)
                    # print(GT_flat.shape)
                    loss = self.criterion(SR_flat, GT_flat)
                    loss = self.dice_loss(SR_flat, GT_flat)
                    epoch_loss += loss.item()
                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()

                    # acc += get_accuracy(SR, GT)
                    # SE += get_sensitivity(SR, GT)
                    # SP += get_specificity(SR, GT)
                    # PC += get_precision(SR, GT)
                    # F1 += get_F1(SR, GT)
                    # JS += get_JS(SR, GT)
                    DC += get_DC(SR_flat, GT_flat)
                    # length += images.size(0)
                    wandb.log({"lr": self.get_lr()})

                # acc = acc/length
                # SE = SE/length
                # SP = SP/length
                # PC = PC/length
                # F1 = F1/length
                # JS = JS/length
                # DC = DC/length
                DC = DC/(i+1)

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] DC: %.4f' % (
                    epoch+1, self.num_epochs,
                    epoch_loss/(i+1), DC))
                wandb.log({"loss/epoch": epoch_loss/(i+1),
                        #    "train/acc": acc,
                        #    "train/sens": SE,
                        #    "train/spec": SP,
                        #    "train/prec": PC,
                        #    "train/f1": F1,
                        #    "train/jacc": JS,
                           "train/dice": DC})
                
                # Decay learning rate
                # if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                #     lr -= (self.lr / float(self.num_epochs_decay))
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = lr
                #     print('Decay learning rate to lr: {}.'.format(lr))

                #===================================== Validation ====================================#
                # self.unet.train(False)
                self.unet.eval()
                with torch.no_grad():
                    epoch_loss = 0.
                    # acc = 0.  # Accuracy
                    # SE = 0.		# Sensitivity (Recall)
                    # SP = 0.		# Specificity
                    # PC = 0. 	# Precision
                    # F1 = 0.		# F1 Score
                    # JS = 0.		# Jaccard Similarity
                    DC = 0.		# Dice Coefficient
                    # length = 0
                    for i, (images, GT) in enumerate(self.valid_loader):
                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = torch.sigmoid(self.unet(images))
                        SR_flat = SR.view(SR.size(0), -1)
                        GT_flat = GT.view(GT.size(0), -1)
                        loss = self.dice_loss(SR_flat, GT_flat)
                        # loss = self.criterion(SR_flat, GT_flat)
                        epoch_loss += loss.item()
                        # acc += get_accuracy(SR, GT)
                        # SE += get_sensitivity(SR, GT)
                        # SP += get_specificity(SR, GT)
                        # PC += get_precision(SR, GT)
                        # F1 += get_F1(SR, GT)
                        # JS += get_JS(SR, GT)
                        DC += get_DC(SR_flat, GT_flat)

                        # length += images.size(0)

                    # acc = acc/length
                    # SE = SE/length
                    # SP = SP/length
                    # PC = PC/length
                    # F1 = F1/length
                    # JS = JS/length
                    DC = DC/(i+1)
                    unet_score = DC
                    self.scheduler.step(DC)

                    print('[Validation] DC: %.4f' % (DC))
                    wandb.log({"loss/epoch_val": epoch_loss/(i+1),
                            #    "val/acc": acc,
                            #    "val/sens": SE,
                            #    "val/spec": SP,
                            #    "val/prec": PC,
                            #    "val/f1": F1,
                            #    "val/jacc": JS,
                               "val/dice": DC})

                    '''
                    torchvision.utils.save_image(images.data.cpu(),
                                                os.path.join(self.result_path,
                                                            '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                    torchvision.utils.save_image(SR.data.cpu(),
                                                os.path.join(self.result_path,
                                                            '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                    torchvision.utils.save_image(GT.data.cpu(),
                                                os.path.join(self.result_path,
                                                            '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                    '''

                    # Save Best U-Net model
                    if unet_score > best_unet_score:
                        best_unet_score = unet_score
                        best_epoch = epoch
                        best_unet = self.unet.state_dict()
                        print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                        torch.save(best_unet, unet_path)

            #===================================== Test ====================================#
            del self.unet
            del best_unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))

            self.unet.eval()
            with torch.no_grad():
                # acc = 0.  # Accuracy
                # SE = 0.		# Sensitivity (Recall)
                # SP = 0.		# Specificity
                # PC = 0. 	# Precision
                # F1 = 0.		# F1 Score
                # JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                # length = 0
                # epoch_loss = 0.
                for i, (images, GT) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = torch.sigmoid(self.unet(images))
                    SR_flat = SR.view(SR.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    # loss = self.dice_loss(SR_flat, GT_flat)
                    # epoch_loss += loss.item()
                    # acc += get_accuracy(SR, GT)
                    # SE += get_sensitivity(SR, GT)
                    # SP += get_specificity(SR, GT)
                    # PC += get_precision(SR, GT)
                    # F1 += get_F1(SR, GT)
                    # JS += get_JS(SR, GT)
                    DC += get_DC(SR_flat, GT_flat)

                    # length += images.size(0)

                # acc = acc/length
                # SE = SE/length
                # SP = SP/length
                # PC = PC/length
                # F1 = F1/length
                # JS = JS/length
                DC = DC/(i+1)

                f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow([self.model_type, best_unet_score, DC, self.lr, best_epoch, self.num_epochs, self.augmentation_prob])
                f.close()
                wandb.log({"test/dice": DC})