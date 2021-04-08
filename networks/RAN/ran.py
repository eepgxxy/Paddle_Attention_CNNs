import paddle
import paddle.nn as nn
from basic_layers import ResidualBlock
from attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar


class ResidualAttentionModel_448input(nn.Layer):
    # for input size 448
    def __init__(self, num_classes):
        super(ResidualAttentionModel_448input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr = False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.mpool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # tbq add
        # 112*112
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128)
        # tbq add end
        self.residual_block1 = ResidualBlock(128, 256, 2)
        # 56*56
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2D(2048),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)

        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.fc(out)

        return out


class ResidualAttentionModel_92(nn.Layer):
    # for input size 224
    def __init__(self, num_classes):
        super(ResidualAttentionModel_92, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr = False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.mpool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2D(2048),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)

        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = paddle.reshape(out, [out.shape[0], -1])

        out = self.fc(out)

        return out


class ResidualAttentionModel_56(nn.Layer):
    # for input size 224
    def __init__(self, num_classes):
        super(ResidualAttentionModel_56, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr = False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.mpool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2D(2048),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)

        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = paddle.reshape(out, [out.shape[0], -1])

        out = self.fc(out)

        return out


class ResidualAttentionModel_92_32input(nn.Layer):
    # for input size 32
    def __init__(self, num_classes):
        super(ResidualAttentionModel_92_32input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=5, stride=1, padding=2, bias_attr=False),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )  # 32*32
        self.mpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128)  # 16*16
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256)  # 8*8 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 4*4
        self.residual_block5 = ResidualBlock(1024, 1024)  # 4*4
        self.residual_block6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=4, stride=1)
        )
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)

        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = paddle.reshape(out, [out.shape[0], -1])

        out = self.fc(out)

        return out


class ResidualAttentionModel_92_32input_update(nn.Layer):
    # for input size 32
    def __init__(self, num_classes):
        super(ResidualAttentionModel_92_32input_update, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )  # 32*32
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=8)
        )
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x):
        out = self.conv1(x)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)

        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)

        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.fc(out)

        return out

if __name__ == '__main__':
    network = ResidualAttentionModel_56(num_classes=10)
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)