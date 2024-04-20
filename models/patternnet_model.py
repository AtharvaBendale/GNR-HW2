import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modified architecture for PatternNet dataset.
"""

class Generator(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.tconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, output_channels, 4, 2, 1, bias=False)

    def forward(self, z):
        # print("Generator Input :", z.shape)
        z = z.squeeze()
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        # print("generator forward :", x.shape)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = torch.tanh(self.tconv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(1024)

        # Adjusted linear layer input size based on output size from conv layers
        # self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        # x = x.view(-1, 512 * 4 * 4)  # Reshape to match the input size of the linear layer
        # x = self.fc(x)
        # print("Discriminator output shape: ", x.shape)  
        return x

# Other components remain unchanged

class DHead(nn.Module):
    def __init__(self, input_channels=1024):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 1, 1)
        # self.fc

    def forward(self, x):
        # print("Dhead input shape: ", x.shape)
        output = torch.sigmoid(self.conv(x))
        output = output.squeeze()
        # print("Dhead output shape: ", output.shape)
        return output

class QHead(nn.Module):
    def __init__(self, input_channels=1024, num_classes=38, num_continuous=2):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, num_classes, 1)
        self.conv_mu = nn.Conv2d(128, num_continuous, 1)
        self.conv_var = nn.Conv2d(128, num_continuous, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
