import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.GAN import DCGAN
from dataset.MNIST import MNIST

from utils import get_img_grid
import os
from tqdm import tqdm

if __name__ == "__main__":
    DATA_DIR = "data/"

    LATENT_DIM = 100
    FEATURES_DIM = 64
    IMG_CHANNELS = 1
    IMG_SIZE = 64

    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.00002
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    SAVE_DIR = 'runs'
    GEN_IMG_SAVE_PATH = os.path.join(SAVE_DIR, 'visualizations')
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'models')
    SAVE_INTERVAL = 100

    os.makedirs(GEN_IMG_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    dataset = MNIST(data_dir=DATA_DIR, image_size=IMG_SIZE, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = DCGAN(latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS,
                features_dim=FEATURES_DIM, device=DEVICE, learning_rate=LEARNING_RATE)

    writer = SummaryWriter('runs/MNIST')
    iteration = 0
    for epoch in range(NUM_EPOCHS):
        img_save_path = os.path.join(
            GEN_IMG_SAVE_PATH, f'epoch_{epoch+1}.jpg')

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            gen_images, g_loss, d_loss = net.train_single_batch(batch)
            writer.add_scalar('Generator Loss per Iteration',
                              g_loss, iteration)
            writer.add_scalar(
                'Discriminator Loss per Iteration', d_loss, iteration)
            iteration += 1

        writer.add_scalar('Generator Loss per Epoch', g_loss, epoch)
        writer.add_scalar('Discriminator Loss per Epoch', d_loss, epoch)
        print(f"EPOCH {epoch}:   ||  G_LOSS: {g_loss}, D_LOSS: {d_loss}")

        get_img_grid(
            gen_images[0:4].detach().cpu(), rows=1, cols=4).save(img_save_path)

        if (epoch) % SAVE_INTERVAL == 0:
            torch.save(net.generator.state_dict(), os.path.join(
                MODEL_SAVE_PATH, "g_latest.pth"))
            torch.save(net.discriminator.state_dict(),
                       os.path.join(MODEL_SAVE_PATH, "d_latest.pth"))
