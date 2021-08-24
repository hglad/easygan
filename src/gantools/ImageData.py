import torch

class ImageData:
    """
    Class intended for use in conjunction with a PyTorch dataloader.
    Labels can be supplied if the image data is labelled.
    """
    def __init__(self, images, labels=None):
        self.x = torch.Tensor(images)
        if labels is None:
            self.y  = torch.ones(images.shape[0]).long()
        else:
            self.y = torch.Tensor(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
