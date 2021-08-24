import torch.nn as nn
import torch

class CDis(nn.Module):
    def __init__(self, n_classes=26, c=64, add_noise=True, noise_magnitude=0.1):
        super(CDis, self).__init__()

        """
        For 90 x 160 x 3 images
        """
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude
        self.labels_to_layer = nn.Sequential(
            nn.Embedding(n_classes, 50),
            nn.Linear(50, 160*90)
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(c, 2*c, kernel_size=(4,5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(2*c, 4*c, kernel_size=(4,5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(4*c, 4*c, kernel_size=(4,5), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(4*c, 8*c, kernel_size=(4,5), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(8*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(8*c, 1, kernel_size=(2,4), stride=1, padding=0, bias=False),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Sigmoid()
            # nn.MaxPool2d(5)
        )

    def added_noise(self, x):
        # noise = (torch.randn(x.shape)*magnitude).to(x.device).detach()
        noise = self.noise_magnitude*torch.randn_like(x).detach()
        return x + noise

    def forward(self, image, labels):
        # print ('DISCRIM FORWARD')
        label_layer = self.labels_to_layer(labels)
        label_layer = label_layer.view((-1, 1, 90, 160))
        # concatenate into input image as a 4th channel
        if self.add_noise:
            x = torch.cat((self.added_noise(image), label_layer), dim=1)
        else:
            x = torch.cat((image, label_layer), dim=1)
        # x = self.added_noise(x)
        # print (x.shape)
        x = self.conv0(x)
        # print ('after conv', x.shape)
        # x = x.view((-1, x.shape[1]*x.shape[2]*x.shape[3]))       # concatenate
        x = x.view((-1, 1))
        # print ('after view', x.shape)
        # print (x.shape)
        return x
