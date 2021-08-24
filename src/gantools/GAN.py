import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import imageio
import torchvision.transforms as transforms
from time import gmtime, strftime, localtime
from torch.utils import data
from .nets.Gen128 import Gen
from .nets.Dis128 import Dis
from .utils import *
from .ImageData import ImageData
from .DiffAugment_pytorch import DiffAugment

class GAN:
    def __init__(self, **cfg_args):
        valid_args = ['batch_size', 'use_cuda', 'epochs', 'lr_g', 'lr_d',
                     'beta1', 'beta2', 'momentum', 'shuffle', 'DiffAugment',
                     'do_plot', 'plot_freq', 'manual_seed', 'loss', 'z_size',
                     'save_gif', 'base_channels', 'add_noise', 'noise_magnitude']

        # Default configuration with same training paramters as original GAN paper

        cfg =    {        # Training parameters
                          'batch_size': 128,
                          'use_cuda': True,
                          'epochs': 100,
                          'lr_g': 0.0002,
                          'lr_d': 0.0002,
                          'beta1': 0.5,
                          'beta2': 0.999,        # = momentum
                          'shuffle': True,       # shuffle data during training
                          'DiffAugment': False,
                          'loss': 'BCELoss',     # loss function

                          # Neural network parameters
                          'z_size': 100,         # latent space vector size
                          'base_channels': 64,   # base number for NN filters
                          'add_noise': True,
                          'noise_magnitude': 0.1,

                          # Plotting parameters
                          'do_plot': True,
                          'plot_freq': 1,

                          # Other
                          'manual_seed': 0,
                          'save_gif': True

                }

        # Apply user-defined properties to configuration dict
        for arg, value in cfg_args.items():
            if arg in valid_args:
                cfg[arg] = value
            else:
                print ("'%s' is not a valid property. See the Readme for a list of properties that can be tweaked." % arg)

        self.cfg = cfg
        if cfg['loss'] == 'BCELoss':
            self.loss = torch.nn.BCELoss(reduction='sum')
        else:
            raise NotImplementedError("Loss function '%s' not implemented. Only 'BCELoss' is implemented so far.")

        self.z_size = cfg['z_size']


    def train_gan(self, imgs, loaded_G=None, loaded_D=None):
        torch.manual_seed(self.cfg['manual_seed'])
        np.random.seed(self.cfg['manual_seed'])
        # torch.autograd.set_detect_anomaly(True)

        try:
            model_folder = self.cfg['model_folder']
        except:
            model_folder = 'models'

        self.imgs = imgs
        self.n_samples = len(imgs)

        epochs = self.cfg['epochs']
        c, h, w = imgs[0].shape
        images_gif = np.zeros((epochs, h, w, c))

        d_error_real = np.zeros(epochs)
        d_error_fake = np.zeros(epochs)
        g_error = np.zeros(epochs)

        # Set generator and discriminator
        self.G = Gen(self.cfg['base_channels'])
        self.D = Dis(self.cfg['base_channels'], self.cfg['add_noise'], self.cfg['noise_magnitude'])

        # Load pre-trained models if supplied
        if loaded_G is not None:
            self.G.load_state_dict(torch.load("%s" % loaded_G))
        else:
            self.G.apply(weights_init)

        if loaded_D is not None:
            self.D.load_state_dict(torch.load("%s" % loaded_D))
        else:
            self.D.apply(weights_init)

        if self.cfg['use_cuda'] == True:
            self.G.to('cuda')
            self.D.to('cuda')

        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg['lr_g'], betas=(self.cfg['beta1'], self.cfg['beta2']))
        D_opt = torch.optim.Adam(self.D.parameters(), lr=self.cfg['lr_d'], betas=(self.cfg['beta1'], self.cfg['beta2']))

        Dataset = ImageData(self.imgs)
        dataloader = data.DataLoader(Dataset, self.cfg['batch_size'], shuffle=self.cfg['shuffle'], num_workers=4, pin_memory=True)
        n_examples = 4

        # Define a latent space vector that remains constant,
        # use to evaluate quality of images during training
        z_verify = torch.randn(1, self.z_size)

        for epoch in range(epochs):
            g_error[epoch], d_error_real[epoch], d_error_fake[epoch] = self.run_epoch(dataloader, G_opt, D_opt)

            sys.stdout.write('epoch: %d/%d, g, d_fake, d_real: %1.4g   %1.4g   %1.4g     \r' % (epoch+1, epochs, g_error[epoch], d_error_fake[epoch], d_error_real[epoch]) )
            sys.stdout.flush()

            if (epoch % self.cfg['plot_freq']) == 0:
                self.G.eval()
                fig, ax = plt.subplots(1, n_examples, figsize=(16,12), sharex=True, sharey=True)
                plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

                for k in range(1, n_examples):
                    if self.cfg['use_cuda']:
                        test_img = self.G( torch.randn(1, self.z_size).cuda())
                    else:
                        test_img = self.G( torch.randn(1, self.z_size))

                    img = un_normalize(test_img[0].permute(1,2,0))
                    ax[k].imshow(img.detach().cpu().numpy())
                    ax[k].set_axis_off()

                # Plot verification image
                if self.cfg['use_cuda']:
                    test_img = self.G( z_verify.cuda())
                else:
                    test_img = self.G( z_verify)

                img = un_normalize(test_img[0].permute(1,2,0)).detach().cpu().numpy()
                images_gif[epoch] = img
                ax[0].imshow(img)
                ax[0].set_axis_off()

                plt.show()
                self.G.train()

        self.G.eval()
        plt.figure()
        plt.plot(d_error_real, label='d error real')
        plt.plot(d_error_fake, label='d error fake')
        plt.plot(g_error, label='g error')
        plt.ylabel('error')
        plt.legend()
        plt.grid()

        # Generate some images using the trained generator
        fig, ax = plt.subplots(4, 4, figsize=(13,8), sharex=True, sharey=True)
        plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.1)

        for k in range(16):
            i = k // 4
            j = k % 4
            if self.cfg['use_cuda']:
                test_img = self.G( torch.randn(1, self.z_size).cuda())
            else:
                test_img = self.G( torch.randn(1, self.z_size))

            img = un_normalize(test_img[0].permute(1,2,0))
            ax[i, j].imshow(img.detach().cpu().numpy())
            ax[i, j].set_axis_off()

        plt.show()
        self.G.train()

        # Save models
        current_time = strftime("%Y-%m-%d-%H%M", localtime())
        os.makedirs("%s/%s" % (model_folder, current_time))
        os.makedirs("results/%s" % current_time)

        name_gen = "generator_%sepochs_%s" % (str(epochs), str(current_time))
        name_dis = "discriminator_%sepochs_%s" % (str(epochs), str(current_time))
        torch.save(self.G.state_dict(), os.path.join(model_folder, current_time, name_gen))
        torch.save(self.D.state_dict(), os.path.join(model_folder, current_time, name_dis))
        print ("Saved generator as %s" % name_gen)
        print ("Saved discriminator as %s" % name_dis)

        if self.cfg['save_gif']:
            images_gif *= 255
            imageio.mimsave('results/%s/progress.gif' % str(current_time), images_gif.astype(np.uint8), fps=10)

        # Save self.cfg dict to txt
        configfile = open('results/%s/cfg.txt' % current_time, 'w')
        configfile.write("---- Configuration ----\n")
        for i, j in self.cfg.items():
            configfile.write("%s: %s\n" % (i,j))

        configfile.close()

    def run_epoch(self, data_loader, G_opt, D_opt):
        g_total_error = 0
        d_total_error_real = 0
        d_total_error_fake = 0

        cuda = self.cfg['use_cuda']
        policy = 'color,translation'

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
            fake_data = self.G(noise(images.size(0), self.z_size, cuda=cuda))

            D_opt.zero_grad()

            # Train on real data
            if augment:
                pred_real = self.D(DiffAugment(real_data, policy))
            else:
                pred_real = self.D(real_data)

            # Calculate error on real data and backpropagate
            error_real = self.loss(pred_real, real_data_target(real_data.size(0), 0.1, cuda))
            (error_real/batch_size).backward()

            # Train on fake data created by generator
            if augment:
                pred_fake = self.D(DiffAugment(fake_data.detach(), policy))
            else:
                pred_fake = self.D(fake_data.detach())

            # Calculate error on fake data and backpropagate
            error_fake = self.loss(pred_fake, fake_data_target(fake_data.size(0), 0, cuda))
            (error_fake/batch_size).backward()
            D_opt.step()

            d_total_error_real += error_real
            d_total_error_fake += error_fake

            """
            Train generator
            """
            G_opt.zero_grad()
            fake_data_gen = self.G(noise(images.size(0), self.z_size, cuda=cuda)) # DO NOT DETACH
            if augment:
                d_on_g_pred = self.D(DiffAugment(fake_data_gen, policy))
            else:
                d_on_g_pred = self.D(fake_data_gen)

            # Calculate error and backpropagate
            g_error = self.loss(d_on_g_pred, real_data_target(d_on_g_pred.size(0), 0, cuda))
            (g_error/batch_size).backward()

            G_opt.step()

            g_total_error += g_error

        # return loss per sample
        return g_total_error/self.n_samples, d_total_error_real/self.n_samples, d_total_error_fake/self.n_samples


    def flush(self):
        torch.cuda.empty_cache()         # free GPU memory
        self.G.to('cpu')
        self.D.to('cpu')
        self.G = None
        self.D = None
