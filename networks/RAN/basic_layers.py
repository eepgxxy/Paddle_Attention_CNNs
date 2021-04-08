import paddle.nn as nn

class ResidualBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2D(input_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2D(input_channels, output_channels//4, 1, 1, bias_attr = False)
        self.bn2 = nn.BatchNorm2D(output_channels//4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(output_channels//4, output_channels//4, 3, stride, padding = 1, bias_attr = False)
        self.bn3 = nn.BatchNorm2D(output_channels//4)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2D(output_channels//4, output_channels, 1, 1, bias_attr = False)
        self.conv4 = nn.Conv2D(input_channels, output_channels , 1, stride, bias_attr = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out