import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import imageio
import glob
import inspect

from time import gmtime, strftime, localtime
from torch.utils import data
from .GAN import GAN
from .nets.Gen128 import Gen128
from .nets.Dis128 import Dis128
from .nets.CGen import CGen
from .nets.CDis import CDis
from .ImageData import ImageData
from .DiffAugment_pytorch import DiffAugment
"""
Not usable yet
"""

class CGAN(GAN):
    def __init__(self, **cfg_args):
        # raise NotImplementedError("To be implemented later")
        GAN.__init__(self)
        self.cgan = True


    def train_cgan(self, imgs, labels, restart=False, **cfg_args):
        torch.manual_seed(self.cfg['manual_seed'])
        np.random.seed(self.cfg['manual_seed'])
        # torch.autograd.set_detect_anomaly(True)

        self.modify_cfg(**cfg_args)

        try:
            model_folder = self.cfg['model_folder']
        except:
            model_folder = 'models'

        self.imgs = imgs
        self.labels = labels

        # if label_names is None:
            # self.label_names = [str(i) for i in labels]
        # else:
            # self.label_names = label_names
        self.label_names = [str(i) for i in labels]

        self.n_samples = len(imgs)
        self.n_labels = np.max(labels) - 1

        epochs = self.cfg['epochs']
        c, h, w = imgs[0].shape

        self.images_gif = np.zeros((epochs+1, h, w, c))

        d_error_real = np.zeros(epochs)
        d_error_fake = np.zeros(epochs)
        g_error = np.zeros(epochs)

        # Initialize weights if model has not been trained/loaded
        if self.has_trained == False or restart == True:
            self.G, self.D = self.set_models()
            self.G.apply(self.weights_init)
            self.D.apply(self.weights_init)
            self.epochs_trained = 0

        if self.cfg['use_cuda'] == True:
            self.G.to('cuda')
            self.D.to('cuda')

        # self.test_model_outputs()

        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg['lr_g'], betas=(self.cfg['beta1'], self.cfg['beta2']))
        D_opt = torch.optim.Adam(self.D.parameters(), lr=self.cfg['lr_d'], betas=(self.cfg['beta1'], self.cfg['beta2']))

        Dataset = ImageData(self.imgs, self.labels)
        dataloader = data.DataLoader(Dataset, self.cfg['batch_size'], shuffle=self.cfg['shuffle'], num_workers=4, pin_memory=True)
        n_examples = 4

        # Define a latent space vector that remains constant,
        # use to evaluate quality of images during training
        z_verify = torch.randn(1, self.cfg['z_size'])
        y_verify = torch.Tensor([0]).long()

        for epoch in range(epochs):
            g_error[epoch], d_error_real[epoch], d_error_fake[epoch] = self.run_epoch_cgan(dataloader, G_opt, D_opt)

            sys.stdout.write('epoch: %d/%d, g, d_fake, d_real: %1.4g   %1.4g   %1.4g     \r' % (epoch+1, epochs, g_error[epoch], d_error_fake[epoch], d_error_real[epoch]) )
            sys.stdout.flush()

            # Show some generated images at given intervals
            if self.cfg['do_plot']:
                if (epoch % self.cfg['plot_interval']) == 0:
                    fig, ax = plt.subplots(1, n_examples, figsize=(16,12), sharex=True, sharey=True)
                    plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

                    for k in range(n_examples):
                        z = torch.randn(1, self.cfg['z_size'])
                        y = torch.randint(0, self.n_labels, size=(1,))
                        img = self.generate_image(z,y)
                        ax[k].imshow(img)
                        ax[k].set_axis_off()

                plt.show()

            # Create verification image for later
            self.images_gif[epoch+1] = self.generate_image(z_verify, y_verify)
            self.epochs_trained += 1

        """
        Done training
        """
        self.has_trained = True

        self.write_results(g_error, d_error_real, d_error_fake)

    def run_epoch_cgan(self, data_loader, G_opt, D_opt):
        g_total_error = 0
        d_total_error_real = 0
        d_total_error_fake = 0

        cuda = self.cfg['use_cuda']
        policy = self.cfg['augment_policy']
        augment = self.cfg['DiffAugment']

        for batch_idx, data_batch in enumerate(data_loader):
            if cuda:
                images = data_batch[0].to('cuda')
                labels = data_batch[1].unsqueeze(1).to('cuda')
            else:
                images = data_batch[0]
                labels = data_batch[1].unsqueeze(1)

            batch_size = images.size(0)
            real_data = images

            """
            Train discriminator
            """
            fake_data = self.G(self.noise(images.size(0), self.cfg['z_size']), labels)

            D_opt.zero_grad()

            # Train on real data
            if augment:
                pred_real = self.D(DiffAugment(real_data, policy), labels)
            else:
                pred_real = self.D(real_data, labels)

            # Calculate error on real data and backpropagate
            error_real = self.L(pred_real, self.real_data_target(real_data.size(0), 0.1))
            (error_real/batch_size).backward()

            # Train on fake data created by generator
            if augment:
                pred_fake = self.D(DiffAugment(fake_data.detach(), policy), labels)
            else:
                pred_fake = self.D(fake_data.detach(), labels)

            # Calculate error on fake data and backpropagate
            error_fake = self.L(pred_fake, self.fake_data_target(fake_data.size(0), 0))
            (error_fake/batch_size).backward()
            D_opt.step()

            d_total_error_real += error_real
            d_total_error_fake += error_fake

            """
            Train generator
            """
            G_opt.zero_grad()
            fake_data_gen = self.G(self.noise(images.size(0), self.cfg['z_size']), labels) # DO NOT DETACH
            if augment:
                d_on_g_pred = self.D(DiffAugment(fake_data_gen, policy), labels)
            else:
                d_on_g_pred = self.D(fake_data_gen, labels)

            # Calculate error and backpropagate
            g_error = self.L(d_on_g_pred, self.real_data_target(d_on_g_pred.size(0), 0))
            (g_error/batch_size).backward()

            G_opt.step()

            g_total_error += g_error

        # return loss per sample
        m = batch_idx + 1       # number of minibatches
        return g_total_error/m, d_total_error_real/m, d_total_error_fake/m
