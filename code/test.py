import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

class Unet(nn.Cell):
    def __init__(self):
        super(Unet, self).__init__()
        # Encoder部分
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, pad_mode='valid')
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode='valid')
        self.relu1_2 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad_mode='valid')
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='valid')
        self.relu2_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, pad_mode='valid')
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='valid')
        self.relu3_2 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, pad_mode='valid')
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode='valid')
        self.relu4_2 = nn.ReLU()
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, pad_mode='valid')
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, pad_mode='valid')
        self.relu5_2 = nn.ReLU()

        # Decoder部分
        self.up_conv_1 = nn.Conv2dTranspose(in_channels=1024, out_channels=512, kernel_size=2, stride=2, pad_mode='valid')

        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, pad_mode='valid')
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, pad_mode='valid')
        self.relu6_2 = nn.ReLU()

        self.up_conv_2 = nn.Conv2dTranspose(in_channels=512, out_channels=256, kernel_size=2, stride=2, pad_mode='valid')

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, pad_mode='valid')
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='valid')
        self.relu7_2 = nn.ReLU()

        self.up_conv_3 = nn.Conv2dTranspose(in_channels=256, out_channels=128, kernel_size=2, stride=2, pad_mode='valid')

        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, pad_mode='valid')
        self.relu8_1 = nn.ReLU()
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='valid')
        self.relu8_2 = nn.ReLU()

        self.up_conv_4 = nn.Conv2dTranspose(in_channels=128, out_channels=64, kernel_size=2, stride=2, pad_mode='valid')

        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, pad_mode='valid')
        self.relu9_1 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode='valid')
        self.relu9_2 = nn.ReLU()

        # 最后的1x1卷积
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, pad_mode='valid')

        # 操作
        self.concat = ops.Concat(axis=1)

    def crop_tensor(self, tensor, target_tensor):
        """
        中心裁剪tensor，使其大小与target_tensor的空间维度一致
        """
        target_size = target_tensor.shape[2]
        tensor_size = tensor.shape[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def construct(self, x):
        # Encoder路径
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)  # 保存用于跳跃连接
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.relu2_2(x4)  # 保存用于跳跃连接
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.relu3_2(x6)  # 保存用于跳跃连接
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.relu4_2(x8)  # 保存用于跳跃连接
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.relu5_2(x10)

        # Decoder路径
        # 第一次上采样
        up1 = self.up_conv_1(x10)  # 上采样
        crop1 = self.crop_tensor(x8, up1)  # 裁剪
        up_1 = self.concat((crop1, up1))  # 拼接

        y1 = self.conv6_1(up_1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        # 第二次上采样
        up2 = self.up_conv_2(y2)
        crop2 = self.crop_tensor(x6, up2)
        up_2 = self.concat((crop2, up2))

        y3 = self.conv7_1(up_2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        # 第三次上采样
        up3 = self.up_conv_3(y4)
        crop3 = self.crop_tensor(x4, up3)
        up_3 = self.concat((crop3, up3))

        y5 = self.conv8_1(up_3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        # 第四次上采样
        up4 = self.up_conv_4(y6)
        crop4 = self.crop_tensor(x2, up4)
        up_4 = self.concat((crop4, up4))

        y7 = self.conv9_1(up_4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        # 最后的1x1卷积
        out = self.conv_10(y8)
        return out

if __name__ == '__main__':
    # 创建输入数据
    input_data = Tensor(np.random.randn(1, 1, 572, 572).astype(np.float32))
    # 实例化模型
    unet = Unet()
    # 前向传播
    output = unet(input_data)
    # 打印输出形状
    print(output.shape)
    # 预期输出形状: (1, 2, 388, 388)
    print(output)