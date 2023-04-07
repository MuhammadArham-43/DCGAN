import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np


class MNIST(Dataset):
    def __init__(self, data_dir="../data", image_size=128, device='cuda') -> None:
        super(MNIST, self).__init__()

        self.transforms = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])

        self.data_dir = data_dir
        self.data = datasets.MNIST(
            root=data_dir, download=True, transform=self.transforms, train=True)
        self.data = self.data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        return img.to(self.device)


if __name__ == "__main__":
    data = MNIST()
    img = data.__getitem__(1)
    img = img.cpu().numpy()
    imshow(np.moveaxis(img, 0, 2))
    plt.show()
