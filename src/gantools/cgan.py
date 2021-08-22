class CGAN:
    def __init__(self, imgs, labels, areas, config):
        self.imgs = imgs
        self.labels = labels
        self.areas = areas
        self.config = config
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.n_samples = len(imgs)
        self.n_labels = np.max(labels) - 1
        self.z_size = 100


    def train_cgan(self, loaded_gen=None, loaded_dis=None):
        # torch.autograd.set_detect_anomaly(True)
        try:
            model_folder = self.config['model_folder']
        except:
            model_folder = 'models'

        torch.manual_seed(self.config['manual_seed'])
        np.random.seed(self.config['manual_seed'])
        epochs = self.config['epochs']

        gen = ConditionalGen()
        dis = ConditionalDisc()
        images_gif = np.zeros((epochs, 90, 160, 3))

        if self.config['use_cuda'] == True:
            dis.to('cuda')
            gen.to('cuda')

        # Load pre-trained models if supplied
        if loaded_gen is not None:
            gen.load_state_dict(torch.load("%s" % loaded_gen, map_location='cpu'))
        else:
            gen.apply(weights_init)

        if loaded_dis is not None:
            dis.load_state_dict(torch.load("%s" % loaded_dis, map_location='cpu'))
        else:
            dis.apply(weights_init)

        d_error_real = np.zeros(epochs)
        d_error_fake = np.zeros(epochs)
        g_error = np.zeros(epochs)

        d_opt = torch.optim.Adam(dis.parameters(), lr=self.config['lr_d'], betas=(self.config['beta1'], self.config['momentum']))
        g_opt = torch.optim.Adam(gen.parameters(), lr=self.config['lr_g'], betas=(self.config['beta1'], self.config['momentum']))

        Dataset = ImageData(self.imgs, self.labels)
        dataloader = data.DataLoader(Dataset, self.config['batch_size'], shuffle=self.config['shuffle'], num_workers=4, pin_memory=True)
        n_examples = 4

        # Define a latent space vector that remains constant,
        # use to evaluate quality of images during training
        z_verify = torch.randn(1, self.z_size)
        y_verify = 16      # oolacile

        for epoch in range(epochs):
            d_error_real[epoch], d_error_fake[epoch], g_error[epoch] = run_epoch_cgan(dis, gen, self.z_size, dataloader, self.config, d_opt, g_opt, self.n_samples)

            sys.stdout.write('epoch: %d/%d, g, d_fake, d_real: %1.4g   %1.4g   %1.4g     \r' % (epoch+1, epochs, g_error[epoch], d_error_fake[epoch], d_error_real[epoch]) )
            sys.stdout.flush()

            if (epoch % self.config['plot_freq']) == 0:
                gen.eval()
                fig, ax = plt.subplots(1, n_examples, figsize=(16,12), sharex=True, sharey=True)
                plt.subplots_adjust(left=0, right=1, top=1, wspace=0.05, hspace=0.05)

                for k in range(1, n_examples):
                    idx = np.random.randint(0, self.n_labels, 1)[0]
                    label = torch.Tensor((idx,)).unsqueeze(1).long()

                    if self.config['use_cuda']:
                        test_img = gen( torch.randn(1, self.z_size).cuda(), label.cuda())
                    else:
                        test_img = gen( torch.randn(1, self.z_size), label)

                    img = un_normalize(test_img[0].permute(1,2,0))
                    ax[k].set_title(areas[idx])
                    ax[k].imshow(img.detach().cpu().numpy())
                    ax[k].set_axis_off()

                # Plot verification image
                label = torch.Tensor((y_verify,)).unsqueeze(1).long()
                if self.config['use_cuda']:
                    test_img = gen( z_verify.cuda(), label.cuda())
                else:
                    test_img = gen( z_verify, label)

                img = un_normalize(test_img[0].permute(1,2,0)).detach().cpu().numpy()
                images_gif[epoch] = img
                ax[0].set_title(areas[y_verify])
                ax[0].imshow(img)
                ax[0].set_axis_off()

                plt.show()
                gen.train()

        gen.eval()
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
            if self.config['use_cuda']:
                test_img = gen( torch.randn(1, self.z_size).cuda(), label.cuda())
            else:
                test_img = gen( torch.randn(1, self.z_size), label)

            img = un_normalize(test_img[0].permute(1,2,0))
            ax[i, j].set_title(self.areas[idx])
            ax[i, j].imshow(img.detach().cpu().numpy())
            ax[i, j].set_axis_off()

        plt.show()
        gen.train()

        # Save models
        current_time = strftime("%Y-%m-%d-%H%M", localtime())
        os.makedirs("%s/%s" % (model_folder, current_time))
        os.makedirs("results/%s" % current_time)

        name_gen = "cgenerator_%sepochs_%s" % (str(epochs), str(current_time))
        name_dis = "cdiscriminator_%sepochs_%s" % (str(epochs), str(current_time))
        torch.save(gen.state_dict(), os.path.join(model_folder, current_time, name_gen))
        torch.save(dis.state_dict(), os.path.join(model_folder, current_time, name_dis))
        print ("Saved generator as %s" % name_gen)
        print ("Saved discriminator as %s" % name_dis)
        images_gif *= 255
        imageio.mimsave('results/%s/progress.gif' % str(current_time), images_gif.astype(np.uint8), fps=10)

        # Save self.config dict to txt
        configfile = open('results/%s/self.config.txt' % current_time, 'w')
        configfile.write("---- self.configuration ----\n")
        for i, j in self.config.items():
            configfile.write("%s: %s\n" % (i,j))

        configfile.close()

    def run_epoch_cgan(self, dis_model, gen_model, gen_input_size, data_loader, d_opt, g_opt, num_samples):
        loss = torch.nn.BCELoss(reduction='sum')
        g_total_error = 0
        d_total_error_real = 0
        d_total_error_fake = 0

        cuda = self.config['use_cuda']
        policy = 'color,translation'
        try:
            augment = self.config['DiffAugment']
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
            fake_data = gen_model(noise(images.size(0), gen_input_size, cuda=cuda), labels)

            d_opt.zero_grad()

            # Train on real data
            if augment:
                pred_real = dis_model(DiffAugment(real_data, policy), labels)
            else:
                pred_real = dis_model(real_data, labels)

            # Calculate error on real data and backpropagate
            error_real = loss(pred_real, real_data_target(real_data.size(0), 0.1, True, cuda))
            (error_real/batch_size).backward()

            # Train on fake data created by generator
            if augment:
                pred_fake = dis_model(DiffAugment(fake_data.detach(), policy), labels)
            else:
                pred_fake = dis_model(fake_data.detach(), labels)

            # Calculate error on fake data and backpropagate
            error_fake = loss(pred_fake, fake_data_target(fake_data.size(0), 0, cuda))
            (error_fake/batch_size).backward()
            d_opt.step()

            d_total_error_real += error_real
            d_total_error_fake += error_fake

            """
            Train generator
            """
            g_opt.zero_grad()
            fake_data_gen = gen_model(noise(images.size(0), gen_input_size, cuda=cuda), labels) # DO NOT DETACH
            if augment:
                d_on_g_pred = dis_model(DiffAugment(fake_data_gen, policy), labels)
            else:
                d_on_g_pred = dis_model(fake_data_gen, labels)

            # Calculate error and backpropagate
            g_error = loss(d_on_g_pred, real_data_target(d_on_g_pred.size(0), 0, False, cuda))
            (g_error/batch_size).backward()

            g_opt.step()

            g_total_error += g_error

        # return loss per sample
        return d_total_error_real/num_samples, d_total_error_fake/num_samples, g_total_error/num_samples
