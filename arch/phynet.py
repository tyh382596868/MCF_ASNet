import torch
from torch.utils.checkpoint import checkpoint






class phyNetModel(torch.nn.Module):
    def __init__(self):
        super(phyNetModel, self).__init__()

        def conv_block(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, (3, 3), stride=(1, 1), padding=1),
                torch.nn.GroupNorm(4, out_channels),  # Replace BatchNorm with GroupNorm
                torch.nn.LeakyReLU()
            )

        def upconv_block(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=1, output_padding=1),
                torch.nn.GroupNorm(4, out_channels),
                torch.nn.LeakyReLU()
            )

        # Downsampling layers
        self.layer_01 = conv_block(1, 32)
        self.layer_01_pool = torch.nn.MaxPool2d((2, 2))

        self.layer_02 = conv_block(32, 64)
        self.layer_02_pool = torch.nn.MaxPool2d((2, 2))

        self.layer_03 = conv_block(64, 128)
        self.layer_03_pool = torch.nn.MaxPool2d((2, 2))

        self.layer_04 = conv_block(128, 256)
        self.layer_04_pool = torch.nn.MaxPool2d((2, 2))

        self.layer_05 = conv_block(256, 512)

        # Upsampling layers
        self.layer_06_up = upconv_block(512, 256)
        self.layer_06 = conv_block(512, 256)

        self.layer_07_up = upconv_block(256, 128)
        self.layer_07 = conv_block(256, 128)

        self.layer_08_up = upconv_block(128, 64)
        self.layer_08 = conv_block(128, 64)

        self.layer_09_up = upconv_block(64, 32)
        self.layer_09 = conv_block(64, 32)

        self.layer_10 = torch.nn.Conv2d(32, 1, (1, 1))

    def forward(self, x):
        # Downsampling path with checkpointing
        x1 = checkpoint(self.layer_01, x)
        x2 = checkpoint(self.layer_02, self.layer_01_pool(x1))
        x3 = checkpoint(self.layer_03, self.layer_02_pool(x2))
        x4 = checkpoint(self.layer_04, self.layer_03_pool(x3))
        x5 = checkpoint(self.layer_05, self.layer_04_pool(x4))

        # Upsampling path
        x6_up = self.layer_06_up(x5)
        x6 = self.layer_06(torch.cat((x6_up, x4), dim=1))
        del x5, x4  # Free memory

        x7_up = self.layer_07_up(x6)
        x7 = self.layer_07(torch.cat((x7_up, x3), dim=1))
        del x6, x3

        x8_up = self.layer_08_up(x7)
        x8 = self.layer_08(torch.cat((x8_up, x2), dim=1))
        del x7, x2

        x9_up = self.layer_09_up(x8)
        x9 = self.layer_09(torch.cat((x9_up, x1), dim=1))
        del x8, x1

        x10 = self.layer_10(x9)
        return x10






if __name__ == "__main__":


    # Testing the network
    def test_network():
        model = phyNetModel()
        model.eval()  # Set the model to evaluation mode

        # Create a random input tensor with batch size 1, 1 channel, and 64x64 image size
        test_input = torch.rand(1, 1, 1920, 2560)

        # Perform a forward pass
        with torch.no_grad():
            test_output = model(test_input)

        # Print the shape of the output
        print(f"Output shape: {test_output.shape}")

    # Run the test
    test_network()