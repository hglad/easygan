import torch

class ImageData:
    def __init__(self, images, labels):
        # input_data = input_data[:][0]
        self.x = torch.Tensor(images)
        self.y = torch.Tensor(labels).long()


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
