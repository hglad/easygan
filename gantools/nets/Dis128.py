import torch.nn as nn
from torch import randn_like

class Dis128(nn.Module):
    def __init__(self, c=64, add_noise=True, noise_magnitude=0.1):
        super(Dis128, self).__init__()

        """
        For 128 x 128 x 3 images (from tutorial)
        """
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(c, 2*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(2*c, 4*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(4*c, 4*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(4*c, 8*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8*c),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.3),

            nn.Conv2d(8*c, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.MaxPool2d(5)
        )

    def added_noise(self, x):
        # noise = (torch.randn(x.shape)*magnitude).to(x.device).detach()
        noise = self.noise_magnitude*randn_like(x).detach()
        return x + noise

    def forward(self, x):
        # print ('DISCRIM FORWARD')
        # print (x.shape)
        if self.add_noise:
            x = self.added_noise(x)

        x = self.conv0(x)
        # print ('after conv', x.shape)
        x = x.view((-1, 1))       # concatenate
        # print ('after view', x.shape)
        return x
