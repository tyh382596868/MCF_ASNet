import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积层"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class conv_block(nn.Module):
    """
    优化卷积块
    使用深度可分离卷积替代标准卷积
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """
    优化上采样卷积块
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    """
    优化版U-Net
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super(U_Net, self).__init__()
        print("Optimized U-Net")
        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[3] * 2, filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[2] * 2, filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[1] * 2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[0] * 2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


if __name__ == "__main__":


    # Testing the network
    def test_network():
        model = U_Net()
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