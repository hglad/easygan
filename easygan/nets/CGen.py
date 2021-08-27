import torch.nn as nn
import torch

class CGen(nn.Module):
    def __init__(self, n_classes=26, c=64):
        super(CGen, self).__init__()

        """
        Generates 90 x 160 x 3 image using labels  (h, w, c)
        """

        self.embed = nn.Sequential(
            nn.Embedding(n_classes, 50),
            nn.Linear(50, 16))

        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(116, 8*c, kernel_size=(6,9), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(8*c, 4*c, kernel_size=(4,6), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(4*c, 2*c, kernel_size=(2,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(2*c, c, kernel_size=(5,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(False),

            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent_vector, labels):     # supply z and y
        y_layer = self.embed(labels).squeeze(1)

        x = torch.cat((latent_vector, y_layer), dim=1)
        # print (x.shape)
        x = x.unsqueeze(2).unsqueeze(2)  # dimensions [batch_size, 116, 1, 1]
        # print (x.shape)
        x = self.conv0(x)                # dimensions [batch_size, 3, 90, 160]
        # print (x.shape)

        return x


"""
# OLD
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(116, 8*c, kernel_size=(6,9), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(8*c, 4*c, kernel_size=(3,5), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(4*c, 2*c, kernel_size=(4,5), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(2*c),
            nn.ReLU(False),

            # nn.ConvTranspose2d(2*c, 2*c, kernel_size=(3,5), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(2*c),
            # nn.LeakyReLU(0.2, inplace=False),
            # nn.ReLU(False),

            nn.ConvTranspose2d(2*c, c, kernel_size=(4,5), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(False),

            nn.ConvTranspose2d(c, 3, kernel_size=(4,4), stride=2, padding=0, bias=False),
            nn.Tanh()
        )
"""
