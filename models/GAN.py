import torch
import torch.nn as nn
from torch.optim import Adam
from torchsummary import summary


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.activation(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_dims) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            img_channels, feature_dims, kernel_size=4, stride=2, padding=1)
        self.reluActivation = nn.LeakyReLU(0.2)

        self.block1 = DiscriminatorBlock(
            feature_dims, feature_dims * 2, 4, 2, 1)
        self.block2 = DiscriminatorBlock(
            feature_dims * 2, feature_dims * 4, 4, 2, 1)
        self.block3 = DiscriminatorBlock(
            feature_dims * 4, feature_dims * 8, 4, 2, 1)

        self.conv2 = nn.Conv2d(feature_dims * 8, 1,
                               kernel_size=4, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reluActivation(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.sigmoid(self.conv2(x))
        # print(x)
        # print("DISC OUTPUT SHAPE: ", x.shape)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(GeneratorBlock, self).__init__()
        self.convTranspose = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.batch_norm(x)
        return self.activation(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, features_dim) -> None:
        super(Generator, self) .__init__()

        self.block1 = GeneratorBlock(latent_dim, features_dim * 16, 4, 1, 0)
        self.block2 = GeneratorBlock(
            features_dim * 16, features_dim * 8, 4, 2, 1)
        self.block3 = GeneratorBlock(
            features_dim * 8, features_dim * 4, 4, 2, 1)
        self.block4 = GeneratorBlock(
            features_dim * 4, features_dim * 2, 4, 2, 1)

        self.convTranspose = nn.ConvTranspose2d(
            features_dim * 2, img_channels, kernel_size=4, stride=2, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.activation(self.convTranspose(x))


class DCGAN(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, features_dim=64, device='cuda', learning_rate=0.0002) -> None:
        super(DCGAN, self).__init__()

        self.device = device
        self.latent_dim = latent_dim

        self.discriminator = Discriminator(
            img_channels=img_channels, feature_dims=features_dim).to(self.device)
        self.generator = Generator(
            img_channels=img_channels, latent_dim=latent_dim, features_dim=features_dim).to(self.device)

        self.g_optim = Adam(self.generator.parameters(),
                            lr=learning_rate, betas=(0.5, 0.999))
        self.d_optim = Adam(self.discriminator.parameters(),
                            lr=learning_rate, betas=(0.5, 0.999))

        self._init_weights(self.discriminator)
        self._init_weights(self.generator)

        # for p in self.discriminator.parameters():
        #     p.register_hook(lambda grad: torch.clamp(
        #         grad, -1, 1))

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.generator(x)

    def _init_weights(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def train_single_batch(self, real_data):
        # print(real_data.size())
        batch_size = real_data.shape[0]

        for _ in range(1):
            generated_images, g_loss = self._train_generator(batch_size)
        for _ in range(1):
            d_loss = self._train_discriminator(real_data)

        return generated_images, g_loss, d_loss

    def _train_generator(self, batch_size):
        self.g_optim.zero_grad()
        noise = torch.randn(
            (batch_size, self.latent_dim, 1, 1)).to(self.device)

        generated_images = self.generator(noise)
        # print(generated_images.shape)
        # print(generated_images[0])
        # print("\n**********************************\n")
        predictions = self.discriminator(generated_images).reshape(-1, 1)
        labels = torch.ones(noise.size(0), 1).to(self.device)
        # print()
        # print(predictions)
        # print(labels)
        # print(generated_images.shape)
        # print(predictions.shape)
        # print(labels.shape)
        # print()
        g_loss = self.criterion(predictions, labels)
        # print("G_LOSS: ", g_loss)
        g_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.g_optim.step()

        return generated_images, g_loss

    def _train_discriminator(self, real_data):
        self.d_optim.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        real_predictions = self.discriminator(real_data).reshape(-1, 1)
        # print("REAL DATA SHAPE: ", real_data.shape)
        # print(real_data[0])
        noise = torch.randn(
            (real_data.shape[0], self.latent_dim, 1, 1)).to(self.device)
        fake_labels = torch.zeros(noise.size(0), 1).to(self.device)
        fake_predictions = self.discriminator(
            self.generator(noise)).reshape(-1, 1)
        # print()
        # print(fake_labels.shape)
        # print(fake_predictions.shape)
        # print(real_labels.shape)
        # print(real_predictions.shape)
        # print()
        real_loss = self.criterion(real_labels, real_predictions)
        fake_loss = self.criterion(fake_labels, fake_predictions)
        # print(real_loss, fake_loss)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.d_optim.step()

        return d_loss


if __name__ == "__main__":
    d = Discriminator(1, 4)
    g = Generator(100, 1, 64)
    # summary(d, (1, 64, 64))
    summary(g, (100, 1, 1))
