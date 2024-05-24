"""
File: image_restoration_model.py
Description: Image Restoration Model (U-NET with 1 Encoder and 2 Decoders)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out += identity
        out = self.silu(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.resblocks = nn.Sequential(
            ResBlock(output_channels),
            ResBlock(output_channels),
            ResBlock(output_channels),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x) -> torch.Tensor:
        """
        Encodes tensor, returning a tuple containing
        the un-pooled tensor as well as the pooled tensor
        """
        x = self.conv(x)
        x = self.resblocks(x)
        x_identity = x
        x_pooled = self.maxpool(x)
        return (x_pooled, x_identity)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eb1 = EncoderBlock(3, 32)
        self.eb2 = EncoderBlock(32, 64)
        self.eb3 = EncoderBlock(64, 128)
        self.eb4 = EncoderBlock(128, 256)
        self.eb5 = EncoderBlock(256, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, res1 = self.eb1(x)
        x, res2 = self.eb2(x)
        x, res3 = self.eb3(x)
        x, res4 = self.eb4(x)
        _, res5 = self.eb5(x)
        return [res1, res2, res3, res4, res5]


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, input_channels // 2, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1, stride=1
        )
        self.silu = nn.SiLU()
        self.resblocks = nn.Sequential(
            ResBlock(output_channels),
            ResBlock(output_channels),
            ResBlock(output_channels),
        )

    def forward(self, x, skip_connection):
        # print(f"x.size() before interpolation: {x.size()}")
        desired_height = skip_connection.size()[2]
        desired_width = skip_connection.size()[3]
        x = F.interpolate(
            x,
            size=(desired_height, desired_width),
            mode="bilinear",
            align_corners=False,
        )
        # print(f"x.size(): {x.size()}")
        x = self.conv1(x)
        # print(f"x.size(): {x.size()}")
        # print(f"skip_connection.size(): {skip_connection.size()}")
        x = torch.cat((x, skip_connection), dim=1)
        # print(f"Concatenated tensor size: {x.size()}")
        x = self.conv2(x)
        x = self.silu(x)
        x = self.resblocks(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.db1 = DecoderBlock(512, 256)
        self.db2 = DecoderBlock(256, 128)
        self.db3 = DecoderBlock(128, 64)
        self.db4 = DecoderBlock(64, 32)
        self.conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, res) -> torch.Tensor:
        x = self.db1(res[4], res[3])
        x = self.db2(x, res[2])
        x = self.db3(x, res[1])
        x = self.db4(x, res[0])
        x = self.conv(x)
        x = torch.tanh(x)
        return x


class BinaryMaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.db1 = DecoderBlock(512, 256)
        self.db2 = DecoderBlock(256, 128)
        self.db3 = DecoderBlock(128, 64)
        self.db4 = DecoderBlock(64, 32)
        self.conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, res) -> torch.Tensor:
        x = self.db1(res[4], res[3])
        x = self.db2(x, res[2])
        x = self.db3(x, res[1])
        x = self.db4(x, res[0])
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x


class ImageRestorationModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        image_decoder: ImageDecoder,
        binary_mask_decoder: BinaryMaskDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.image_decoder = image_decoder
        self.binary_mask_decoder = binary_mask_decoder

    def forward(
        self, corrupted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Image Restoration Model.

        Given a `corrupted_image` with shape (B, C, H, W) where B = batch size, C = # channels,
        H = image height, W = image width and normalized values between -1 and 1,
        run the Image Restoration Model forward and return a tuple of two tensors:
        (`predicted_image`, `predicted_binary_mask`).

        The `predicted_image` should be the output of the Image Decoder (B, C, H, W). In the
        assignment this is referred to as x^{hat}. This is NOT the `reconstructed_image`,
        referred to as `x_{reconstructed}` in the assignment handout.

        The `predicted_binary_mask` should be the output of the Binary Mask Decoder (B, 1, H, W). This
        is `m^{hat}` in the assignment handout.
        """
        encoded_image = self.encoder(corrupted_image)
        predicted_binary_mask = self.binary_mask_decoder(encoded_image)
        predicted_image = self.image_decoder(encoded_image)
        return (predicted_image, predicted_binary_mask)


def create_model():
    encoder = Encoder()
    image_decoder = ImageDecoder()
    binary_mask_decoder = BinaryMaskDecoder()
    model = ImageRestorationModel(
        encoder=encoder,
        image_decoder=image_decoder,
        binary_mask_decoder=binary_mask_decoder,
    )
    return model
