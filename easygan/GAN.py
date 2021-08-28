import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import imageio
import glob
import torchvision.transforms as transforms
from time import gmtime, strftime, localtime
from torch.utils import data
from .nets.Gen128 import Gen128
from .nets.Dis128 import Dis128
from .ImageData import ImageData
from .DiffAugment_pytorch import DiffAugment

class GAN:
    """
    Class for training a GAN with functionality to tweak training parameters and
    more. The user only needs to supply the image data to start training the
    generator (G) and discriminator (D). When finished training, the models are
    automatically saved along with some generated images.

    Implementation of generator and discriminator is mostly inspired by
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html ,

    which is again inspired by the original GAN paper
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    """

    def __init__(self):
        """
        Initialize default parameters and flags. Takes no arguments (yet).
        """

        self.valid_args = ['batch_size', 'use_cuda', 'epochs', 'lr_g', 'lr_d',
                     'beta1', 'beta2', 'shuffle', 'DiffAugment',
                     'do_plot', 'plot_interval', 'manual_seed', 'loss', 'z_size',
                     'save_gif', 'base_channels', 'add_noise', 'noise_magnitude',
                     'augment_policy', 'model_folder', 'custom_G', 'custom_D']

        # Default configuration with same training paramters as original GAN paper
        self.cfg =    {   # Training parameters
                          'batch_size': 128,
                          'use_cuda': True,
                          'epochs': 100,
                          'lr_g': 0.0002,
                          'lr_d': 0.0002,
                          'beta1': 0.5,
                          'beta2': 0.999,        # = momentum
                          'shuffle': True,       # shuffle data during training
                          'loss': 'BCELoss',     # loss function
                          'DiffAugment': False,
                          'augment_policy': 'color,translation',  # augmentation methods if DiffAugment is enabled

                          # Neural network parameters
                          'z_size': 100,         # latent space vector size
                          'base_channels': 64,   # base number for NN filters
                          'add_noise': True,
                          'noise_magnitude': 0.1,# magnitude of added noise to D
                          'custom_G': None,      # to be implemented
                          'custom_D': None,

                          # Plotting parameters
                          'do_plot': True,       # plot generated images during training
                          'plot_interval': 1,    # images are plotted every n epochs

                          # Other
                          'manual_seed': 0,
                          'save_gif': True,
                          'model_folder': 'models'

                }

        self.has_trained = False

    def modify_cfg(self, **cfg_args):
        """
        Apply user-defined properties to configuration dict. Automatically run
        when supplying additional arguments to self.train_gan.
        """

        for arg, value in cfg_args.items():
            if arg in self.valid_args:
                self.cfg[arg] = value
            else:
                print ("'%s' is not a valid property. See the Readme for a list of properties that can be tweaked." % arg)

        if self.cfg['loss'] == 'BCELoss':
            self.L = torch.nn.BCELoss(reduction='mean')
        else:
            raise NotImplementedError("Loss function '%s' not implemented. Only 'BCELoss' is implemented so far.")

        # Check that supplied properties are valid
        intargs = ['batch_size', 'epochs', 'manual_seed', 'z_size', 'base_channels', 'plot_interval']
        floatargs = ['lr_g', 'lr_d', 'beta1', 'beta2', 'noise_magnitude']
        boolargs = ['use_cuda', 'DiffAugment', 'do_plot', 'add_noise', 'save_gif', 'shuffle']
        stringargs = ['augment_policy', 'model_folder']

        for arg in intargs:
            if not isinstance(self.cfg[arg], int):
                raise ValueError("%s must be an int, not %s" % (arg, type(arg)))

        for arg in floatargs:
            if not isinstance(self.cfg[arg], float):
                raise ValueError("%s must be a float, not %s" % (arg, type(arg)))

        for arg in boolargs:
            if not isinstance(self.cfg[arg], bool):
                raise ValueError("%s must be a bool, not %s" % (arg, type(arg)))

        for arg in stringargs:
            if not isinstance(self.cfg[arg], str):
                raise ValueError("%s must be a string, not %s" % (arg, type(arg)))

    def set_models(self):
        """
        Helper function for defining G and D in the class environment.
        """

        if self.cfg['custom_G'] is None:
            G = Gen128(self.cfg['base_channels'])
        else:
            raise NotImplementedError('Custom generator is not supported yet')

        if self.cfg['custom_D'] is None:
            D = Dis128(self.cfg['base_channels'], self.cfg['add_noise'], self.cfg['noise_magnitude'])
        else:
            raise NotImplementedError('Custom discriminator is not supported yet')

        return G, D

    def load_state(self, timestamp):
        """
        Load a saved generator and discriminator from disk. May be a better idea
        to instead require individual paths to G and D.

        Args:
            timestamp: str, name of folder containing G and D.
        """

        folder = self.cfg['model_folder']
        G_path = glob.glob(os.path.join(folder, timestamp, '*generator*'))
        D_path = glob.glob(os.path.join(folder, timestamp, '*discriminator*'))
        if len(G_path) == 0:
            raise FileNotFoundError("Could not find a generator model in %s. Make sure that the specified folder is correct." % os.path.join(folder, timestamp))

        if len(D_path) == 0:
            raise FileNotFoundError("Could not find a discriminator model in %s. Make sure that the specified folder is correct." % os.path.join(folder, timestamp))

        # Set generator and discriminator
        self.G, self.D = self.set_models()

        self.G.load_state_dict(torch.load(G_path[0], map_location='cpu'), strict=False)
        self.D.load_state_dict(torch.load(D_path[0], map_location='cpu'), strict=False)
        if self.cfg['use_cuda']:
            self.G.to('cuda')
            self.D.to('cuda')

        self.has_trained = True

    def train_gan(self, imgs, restart=False, **cfg_args):
        """
        Train the generator (G) and discriminator (D) using provided images.
        Outputs generated images during training by default, and saves G and D
        after training is done.

        Args:
            imgs: Tensor, with dimensions (num_samples, 3, height, width).
            restart: bool, re-initializes weights for G and D if True.
                     Effectively resets G and D to untrained states.
            **cfg_args: additional properties that the user can provide. See
                        Readme for a list of properties.
        """

        # torch.autograd.set_detect_anomaly(True)
        self.modify_cfg(**cfg_args)

        torch.manual_seed(self.cfg['manual_seed'])
        np.random.seed(self.cfg['manual_seed'])

        self.imgs = imgs
        self.n_samples = len(imgs)
        self.z_size = self.cfg['z_size']
        model_folder = self.cfg['model_folder']
        epochs = self.cfg['epochs']

        c, h, w = imgs[0].shape
        images_gif = np.zeros((epochs+1, h, w, c))

        d_error_real = np.zeros(epochs)
        d_error_fake = np.zeros(epochs)
        g_error = np.zeros(epochs)

        # Initialize weights if model has not been trained/loaded
        if self.has_trained == False or restart == True:
            self.G, self.D = self.set_models()
            self.G.apply(self.weights_init)
            self.D.apply(self.weights_init)

        if self.cfg['use_cuda'] == True:
            self.G.to('cuda')
            self.D.to('cuda')

        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg['lr_g'], betas=(self.cfg['beta1'], self.cfg['beta2']))
        D_opt = torch.optim.Adam(self.D.parameters(), lr=self.cfg['lr_d'], betas=(self.cfg['beta1'], self.cfg['beta2']))

        Dataset = ImageData(self.imgs)
        dataloader = data.DataLoader(Dataset, self.cfg['batch_size'], shuffle=self.cfg['shuffle'], num_workers=4, pin_memory=True)
        n_examples = 4

        # Define a latent space vector that remains constant,
        # use to show evolution of latent space --> image over time
        z_verify = torch.randn(1, self.z_size)
        images_gif[0] = self.generate_image(z_verify)   # should look like noise if model is untrained

        for epoch in range(epochs):
            g_error[epoch], d_error_real[epoch], d_error_fake[epoch] = self.run_epoch(dataloader, G_opt, D_opt)

            sys.stdout.write('epoch: %d/%d, g, d_fake, d_real: %1.4g   %1.4g   %1.4g     \r' % (epoch+1, epochs, g_error[epoch], d_error_fake[epoch], d_error_real[epoch]) )
            sys.stdout.flush()

            # Show some generated images at given intervals
            if (epoch % self.cfg['plot_interval']) == 0:
                fig, ax = plt.subplots(1, n_examples, figsize=(16,12), sharex=True, sharey=True)
                plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

                for k in range(n_examples):
                    z = torch.randn(1, self.z_size)
                    img = self.generate_image(z)
                    ax[k].imshow(img)
                    ax[k].set_axis_off()

            # Plot verification image
            images_gif[epoch+1] = self.generate_image(z_verify)
            plt.show()
        """
        Done training
        """
        self.has_trained = True

        # Create folders for results and models with timestamp
        current_time = strftime("%Y-%m-%d-%H%M", localtime())
        os.makedirs("%s/%s" % (model_folder, current_time))
        os.makedirs("results/%s" % current_time)

        # Show some results and save models
        lossplot = plt.figure()
        plt.plot(d_error_real, label='d error real')
        plt.plot(d_error_fake, label='d error fake')
        plt.plot(g_error, label='g error')
        plt.ylabel('error')
        plt.legend()
        plt.grid()
        lossplot.savefig('results/%s/loss.png' % current_time)

        # Generate some images using the trained generator
        fig, ax = plt.subplots(4, 4, figsize=(16,18), sharex=True, sharey=True)
        plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

        for k in range(16):
            i = k // 4
            j = k % 4
            z = torch.randn(1, self.z_size)
            img = self.generate_image(z)
            ax[i, j].imshow(img)
            ax[i, j].set_axis_off()

        fig.savefig('results/%s/4x4_generated.png' % current_time)
        plt.show()

        name_gen = "generator_%sepochs_%s" % (str(epochs), str(current_time))
        name_dis = "discriminator_%sepochs_%s" % (str(epochs), str(current_time))
        torch.save(self.G.state_dict(), os.path.join(model_folder, current_time, name_gen))
        torch.save(self.D.state_dict(), os.path.join(model_folder, current_time, name_dis))
        print ("Saved generator as %s" % name_gen)
        print ("Saved discriminator as %s" % name_dis)

        if self.cfg['save_gif']:
            images_gif *= 255
            imageio.mimsave('results/%s/progress.gif' % str(current_time), images_gif.astype(np.uint8), fps=10)

        # Write self.cfg dict items to txt
        configfile = open('results/%s/cfg.txt' % current_time, 'w')
        configfile.write("---- Configuration ----\n")
        for i, j in self.cfg.items():
            configfile.write("%s: %s\n" % (i,j))

        configfile.close()

    def run_epoch(self, data_loader, G_opt, D_opt):
        """
        Use the provided dataset to train G and D. A single epoch has been run
        when every image has been used once, and the models have updated their
        parameters accordingly. Function is used by self.train_gan.

        Args:
            data_loader: torch.utils.data.DataLoader, used for easy handling of
                         training data.
            G_opt: torch.optim, PyTorch optimizer for optimizing weights of G.
            D_opt: torch.optim, PyTorch optimizer for optimizing weights of D.
        Returns:
            g_error, d_real_error, d_fake_error: floats, average loss per sample
                   for generator and discriminator.

        """
        g_total_error = 0
        d_total_error_real = 0
        d_total_error_fake = 0
        policy = self.cfg['augment_policy']

        for batch_idx, data_batch in enumerate(data_loader):
            if self.cfg['use_cuda']:
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
            fake_data = self.G(self.noise(images.size(0), self.z_size))

            D_opt.zero_grad()

            # Train on real data
            if self.cfg['DiffAugment']:
                pred_real = self.D(DiffAugment(real_data, policy))
            else:
                pred_real = self.D(real_data)

            # Calculate error on real data and backpropagate
            error_real = self.L(pred_real, self.real_data_target(real_data.size(0), 0.1))
            error_real.backward()

            # Train on fake data created by generator
            if self.cfg['DiffAugment']:
                pred_fake = self.D(DiffAugment(fake_data.detach(), policy))
            else:
                pred_fake = self.D(fake_data.detach())

            # Calculate error on fake data and backpropagate
            error_fake = self.L(pred_fake, self.fake_data_target(fake_data.size(0), 0))
            error_fake.backward()
            D_opt.step()

            d_total_error_real += error_real
            d_total_error_fake += error_fake

            """
            Train generator
            """
            G_opt.zero_grad()
            fake_data_gen = self.G(self.noise(images.size(0), self.z_size))
            if self.cfg['DiffAugment']:
                d_on_g_pred = self.D(DiffAugment(fake_data_gen, policy))
            else:
                d_on_g_pred = self.D(fake_data_gen)

            # Calculate error and backpropagate
            g_error = self.L(d_on_g_pred, self.real_data_target(d_on_g_pred.size(0), 0))
            g_error.backward()

            G_opt.step()

            g_total_error += g_error

        # return loss per sample
        m = batch_idx + 1       # number of minibatches
        return g_total_error/m, d_total_error_real/m, d_total_error_fake/m

    def generate_image(self, z):
        """
        Use generator to create an image using latent space vector z.

        Args:
            z: torch.Tensor, of dimensions (1, 100) representing latent space
               vector.
        Returns:
            img: np.ndarray, of dimensions (128, 128, 3).
        """

        self.G.eval()
        if self.cfg['use_cuda']:
            img_tn = self.G(z.cuda())
        else:
            img_tn = self.G(z)

        img = self.un_normalize(img_tn[0].permute(1,2,0)).detach().cpu().numpy()
        self.G.train()
        return img

    def real_data_target(self, size, delta=0):
        '''
        Tensor containing ones, representing labels for real data.
        '''
        data = torch.ones(size, 1) - delta
        if self.cfg['use_cuda']:
            return data.cuda()
        return data

    def fake_data_target(self, size, delta=0):
        '''
        Tensor containing zeros, representing labels for fake data.
        '''
        data = torch.zeros(size, 1) + delta
        if self.cfg['use_cuda']:
            return data.cuda()
        return data

    def noise(self, size, n):
        """
        Create latent vector for generator input.
        """
        noise = torch.randn(size, n)
        if self.cfg['use_cuda']:
            return noise.cuda()
        return noise

    def weights_init(self, m):
        """
        Initialize model weights with mean of 0 and standard deviation 0.02.
        """
        classname = m.__class__.__name__

        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def normalize(self, img, mean=0.5, std=0.5):
        return (img - mean)/std

    def un_normalize(self, img, mean=0.5, std=0.5):
        return mean + img*std

    def flush(self):
        torch.cuda.empty_cache()         # free GPU memory
        # implement procedure for freeing more GPU memory later, seems like
        # PyTorch is holding on to a lot of memory after finishing training
