import torch.nn as nn

class Gen128(nn.Module):
    def __init__(self, c=64):
        super(Gen128, self).__init__()
        """
        Generates 128 x 128 x 3 image     (h, w, c)
        """
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(100, 8*c, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(8*c, 4*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*c),
            nn.ReLU(False),

            nn.ConvTranspose2d(4*c, 2*c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*c),
            nn.ReLU(False),

            # nn.ConvTranspose2d(2*c, 2*c, kernel_size=(3,5), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(2*c),
            # nn.LeakyReLU(0.2, inplace=False),
            # nn.ReLU(False),

            nn.ConvTranspose2d(2*c, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(False),

            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print ('GEN FORWARD')
        x = x.unsqueeze(2).unsqueeze(2)
        # print (x.shape)
        x = self.conv0(x)
        # print (x.shape)
        # x = x.view((-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # x = self.out(x)
        # exit()

        return x
