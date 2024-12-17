import torch
from torch import nn

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_const = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2 --> H/2, W/2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_const = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2 --> H/4, W/4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_const = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2 --> H/8, W/8

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_const = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2 --> H/16, W/16

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_const = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2 --> H/32, W/32


        # Decoder

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1_const = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_deconv1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_const = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn_deconv2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3_const = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_deconv3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4_const = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_deconv4 = nn.BatchNorm2d(32)

        self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5_const = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_deconv5 = nn.BatchNorm2d(32)

        # Final convolution layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.conv1_const(x1)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.conv2_const(x2)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.conv3_const(x3)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.pool3(x3)

        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        x4 = self.bn4(x4)
        x4 = self.conv4_const(x4)
        x4 = self.relu(x4)
        x4 = self.bn4(x4)
        x4 = self.pool4(x4)

        x5 = self.conv5(x4)
        x5 = self.relu(x5)
        x5 = self.bn5(x5)
        x5 = self.conv5_const(x5)
        x5 = self.relu(x5)
        x5 = self.bn5(x5)
        x5 = self.pool5(x5)


        # Decoder

        x = self.deconv1(x5)
        x = self.relu(x)
        x = self.bn_deconv1(x)
        x = self.deconv1_const(x)
        x = self.relu(x)
        x = self.bn_deconv1(x)
        x = x + x4              # Skip connection 1

        x = self.deconv2(x)
        x = self.relu(x)
        x = self.bn_deconv2(x)
        x = self.deconv2_const(x)
        x = self.relu(x)
        x = self.bn_deconv2(x)
        x = x + x3              # Skip connection 2

        x = self.deconv3(x)
        x = self.relu(x)
        x = self.bn_deconv3(x)
        x = self.deconv3_const(x)
        x = self.relu(x)
        x = self.bn_deconv3(x)
        x = x + x2              # Skip connection 3

        x = self.deconv4(x)
        x = self.relu(x)
        x = self.bn_deconv4(x)
        x = self.deconv4_const(x)
        x = self.relu(x)
        x = self.bn_deconv4(x)
        x = x + x1              # Skip connection 4

        x = self.deconv5(x)
        x = self.relu(x)
        x = self.bn_deconv5(x)
        x = self.deconv5_const(x)
        x = self.relu(x)
        x = self.bn_deconv5(x)

        # Final convolution layer
        x = self.final_conv(x)

        return x


class UNet(nn.Module):
    '''
    Taken from: https://www.kaggle.com/code/gokulkarthik/image-segmentation-with-unet-pytorch
    '''
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                               output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):
        contracting_11_out = self.contracting_11(X)  # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out)  # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out)  # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out)  # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out)  # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out)  # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out)  # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out)  # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out)  # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out)  # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(
            torch.cat((expansive_11_out, contracting_41_out), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out)  # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(
            torch.cat((expansive_21_out, contracting_31_out), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out)  # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(
            torch.cat((expansive_31_out, contracting_21_out), dim=1))  # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out)  # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(
            torch.cat((expansive_41_out, contracting_11_out), dim=1))  # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out)  # [-1, num_classes, 256, 256]
        return output_out


