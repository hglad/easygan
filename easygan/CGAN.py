from .GAN import GAN
from .nets.CGen import CGen
from .nets.CDis import CDis
from torch.utils import data
from time import gmtime, strftime, localtime

class CGAN(GAN):
    def __init__(self, **cfg_args):
        GAN.__init__(self, **cfg_args)

    def train_cgan(self, imgs, labels, label_names=None, loaded_G=None, loaded_D=None):
        torch.manual_seed(self.cfg['manual_seed'])
        np.random.seed(self.cfg['manual_seed'])
        # torch.autograd.set_detect_anomaly(True)

        try:
            model_folder = self.cfg['model_folder']
        except:
            model_folder = 'models'

        self.imgs = imgs
        self.labels = labels

        if label_names is None:
            self.label_names = [str(i) for i in labels]
        else:
            self.label_names = label_names

        self.n_samples = len(imgs)
        self.n_labels = np.max(labels) - 1

        epochs = self.cfg['epochs']

        images_gif = np.zeros((epochs, 90, 160, 3))

        d_error_real = np.zeros(epochs)
        d_error_fake = np.zeros(epochs)
        g_error = np.zeros(epochs)

        # Set generator and discriminator
        self.G = CGen(self.n_labels, self.cfg['base_channels'])
        self.D = CDis(self.n_labels, self.cfg['base_channels'], self.cfg['add_noise'], self.cfg['noise_magnitude'])

        # Load pre-trained models if supplied
        if loaded_gen is not None:
            self.G.load_state_dict(torch.load("%s" % loaded_gen))
        else:
            self.G.apply(weights_init)

        if loaded_dis is not None:
            self.D.load_state_dict(torch.load("%s" % loaded_dis))
        else:
            self.D.apply(weights_init)

        if self.cfg['use_cuda'] == True:
            self.G.to('cuda')
            self.D.to('cuda')

        D_opt = torch.optim.Adam(self.D.parameters(), lr=self.cfg['lr_d'], betas=(self.cfg['beta1'], self.cfg['beta2']))
        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg['lr_g'], betas=(self.cfg['beta1'], self.cfg['beta2']))

        Dataset = ImageData(self.imgs, self.labels)
        dataloader = data.DataLoader(Dataset, self.cfg['batch_size'], shuffle=self.cfg['shuffle'], num_workers=4, pin_memory=True)
        n_examples = 4

        # Define a latent space vector that remains constant,
        # use to evaluate quality of images during training
        z_verify = torch.randn(1, self.z_size)
        y_verify = 0

        for epoch in range(epochs):
            g_error[epoch], d_error_real[epoch], d_error_fake[epoch] = run_epoch_cgan(self.G, self.D, self.z_size, dataloader, self.cfg, g_opt, d_opt, self.n_samples)

            sys.stdout.write('epoch: %d/%d, g, d_fake, d_real: %1.4g   %1.4g   %1.4g     \r' % (epoch+1, epochs, g_error[epoch], d_error_fake[epoch], d_error_real[epoch]) )
            sys.stdout.flush()

            if (epoch % self.cfg['plot_freq']) == 0:
                self.G.eval()
                fig, ax = plt.subplots(1, n_examples, figsize=(16,12), sharex=True, sharey=True)
                plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

                for k in range(1, n_examples):
                    idx = np.random.randint(0, self.n_labels, 1)[0]
                    label = torch.Tensor((idx,)).unsqueeze(1).long()

                    if self.cfg['use_cuda']:
                        test_img = self.G( torch.randn(1, self.z_size).cuda(), label.cuda())
                    else:
                        test_img = self.G( torch.randn(1, self.z_size), label)

                    img = un_normalize(test_img[0].permute(1,2,0))
                    ax[k].set_title(self.label_names[idx])
                    ax[k].imshow(img.detach().cpu().numpy())
                    ax[k].set_axis_off()

                # Plot verification image
                label = torch.Tensor((y_verify,)).unsqueeze(1).long()
                if self.cfg['use_cuda']:
                    test_img = self.G( z_verify.cuda(), label.cuda())
                else:
                    test_img = self.G( z_verify, label)

                img = un_normalize(test_img[0].permute(1,2,0)).detach().cpu().numpy()
                images_gif[epoch] = img
                ax[0].set_title(self.label_names[y_verify])
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
            idx = np.random.randint(0, self.n_labels, 1)[0]
            label = torch.Tensor((idx,)).unsqueeze(1).long()
            if self.cfg['use_cuda']:
                test_img = self.G( torch.randn(1, self.z_size).cuda(), label.cuda())
            else:
                test_img = self.G( torch.randn(1, self.z_size), label)

            img = un_normalize(test_img[0].permute(1,2,0))
            ax[i, j].set_title(self.label_names[idx])
            ax[i, j].imshow(img.detach().cpu().numpy())
            ax[i, j].set_axis_off()

        plt.show()
        self.G.train()

        # Save models
        current_time = strftime("%Y-%m-%d-%H%M", localtime())
        os.makedirs("%s/%s" % (model_folder, current_time))
        os.makedirs("results/%s" % current_time)

        name_gen = "cgenerator_%sepochs_%s" % (str(epochs), str(current_time))
        name_dis = "cdiscriminator_%sepochs_%s" % (str(epochs), str(current_time))
        torch.save(self.G.state_dict(), os.path.join(model_folder, current_time, name_gen))
        torch.save(self.D.state_dict(), os.path.join(model_folder, current_time, name_dis))
        print ("Saved generator as %s" % name_gen)
        print ("Saved discriminator as %s" % name_dis)

        if cfg['save_gif']:
            images_gif *= 255
            imageio.mimsave('results/%s/progress.gif' % str(current_time), images_gif.astype(np.uint8), fps=10)

        # Save self.cfg dict to txt
        configfile = open('results/%s/cfg.txt' % current_time, 'w')
        configfile.write("---- Configuration ----\n")
        for i, j in self.cfg.items():
            configfile.write("%s: %s\n" % (i,j))

        configfile.close()

    def run_epoch_cgan(self, data_loader, G_opt, D_opt):
        g_total_error = 0
        d_total_error_real = 0
        d_total_error_fake = 0

        cuda = self.cfg['use_cuda']
        policy = 'color,translation'
        try:
            augment = self.cfg['DiffAugment']
        except:
            augment = False

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
            fake_data = self.G(noise(images.size(0), self.z_size, cuda=cuda), labels)

            d_opt.zero_grad()

            # Train on real data
            if augment:
                pred_real = self.D(DiffAugment(real_data, policy), labels)
            else:
                pred_real = self.D(real_data, labels)

            # Calculate error on real data and backpropagate
            error_real = loss(pred_real, real_data_target(real_data.size(0), 0.1, True, cuda))
            (error_real/batch_size).backward()

            # Train on fake data created by generator
            if augment:
                pred_fake = self.D(DiffAugment(fake_data.detach(), policy), labels)
            else:
                pred_fake = self.D(fake_data.detach(), labels)

            # Calculate error on fake data and backpropagate
            error_fake = loss(pred_fake, fake_data_target(fake_data.size(0), 0, cuda))
            (error_fake/batch_size).backward()
            D_opt.step()

            d_total_error_real += error_real
            d_total_error_fake += error_fake

            """
            Train generator
            """
            G_opt.zero_grad()
            fake_data_gen = self.G(noise(images.size(0), self.z_size, cuda=cuda), labels) # DO NOT DETACH
            if augment:
                d_on_g_pred = self.D(DiffAugment(fake_data_gen, policy), labels)
            else:
                d_on_g_pred = self.D(fake_data_gen, labels)

            # Calculate error and backpropagate
            g_error = loss(d_on_g_pred, real_data_target(d_on_g_pred.size(0), 0, False, cuda))
            (g_error/batch_size).backward()

            G_opt.step()

            g_total_error += g_error

        # return loss per sample
        return g_total_error/self.n_samples, d_total_error_real/self.n_samples, d_total_error_fake/self.n_samples
