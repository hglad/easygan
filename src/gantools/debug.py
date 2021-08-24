from cgan import CGAN
from utils import *


gan = CGAN(epochs=100, batch_size=32, lr_g=0.003, test=True)
