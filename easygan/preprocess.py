import torch
import torchvision.transforms as transforms
import numpy as np

def preprocess(imgs, h=128, w=128, channels_first=False):
    """
    Transforms input data (list of images or array of images) so that pixel
    values are in the domain [-1, 1], resizes and returns a tensor with the
    transformed data.

    Args:
        imgs: list or NumPy array, representing input data to GAN. Images must
            have the shape (height, width, 3) if channels_first is False.
            If channels_first is True, images must have shape (3, height, width)
        h: int, desired image height
        w: int, desired image width
        channels_first: bool, set to True if input images are on the shape
            (3, height, width)
    Returns:
        imgs_tn: Tensor, transformed version of imgs
    """

    if not isinstance(imgs, (list, np.ndarray)):
        raise TypeError('Data must be a list or a NumPy array')

    if isinstance(imgs, np.ndarray):
        n = imgs.shape[0]
    else:
        n = len(imgs)

    test_img = imgs[0]
    if channels_first:
        c, h, w = test_img.shape
    else:
        h, w, c = test_img.shape

    if c != 3:
        raise ValueError("Images must have 3 channels, but %d were found in input image with shape %s" % (c, test_img.shape))

    # Channels first for tensor
    imgs_tn = torch.Tensor(np.zeros((n, 3, h, w), dtype=np.float32))

    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((h,w)),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ])

    # Apply transformation to each provided image
    for i in range(n):
        imgs_tn[i] = transform(imgs[i].copy())

    return imgs_tn
