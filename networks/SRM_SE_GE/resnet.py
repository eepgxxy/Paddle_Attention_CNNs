import paddle
import paddle.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, rclb_layer=None, layer_idx=1, is_ge=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

        if rclb_layer == None:
            self.rclb = None
        else:
            if is_ge:
                self.rclb = rclb_layer(planes, layer_idx)
            else:
                self.rclb = rclb_layer(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.rclb != None:
            out = self.rclb(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rclb_layer=None, layer_idx=1, is_ge=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        if rclb_layer == None:
            self.rclb = None
        else:
            if is_ge:
                self.rclb = rclb_layer(planes * 4, layer_idx)
            else:
                self.rclb = rclb_layer(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.rclb != None:
            out = self.rclb(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, layers, input_channels=3, num_classes=1000, recalibration_type=None):
        super(ResNet, self).__init__()

        
        self.is_ge = True if recalibration_type == 'ge' else False

        if recalibration_type == None:
            self.rclb_layer = None
        elif recalibration_type == 'srm':
            from recalibration_modules import SRMLayer as rclb_layer
            self.rclb_layer = rclb_layer
        elif recalibration_type == 'se':
            from recalibration_modules import SELayer as rclb_layer 
            self.rclb_layer = rclb_layer
        elif recalibration_type == 'ge':
            from recalibration_modules import GELayer as rclb_layer 
            self.rclb_layer = rclb_layer
        else:
            raise NotImplementedError
        
        self.inplanes = 64
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer_idx=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer_idx=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer_idx=4)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0, math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def _make_layer(self, block, planes, blocks, stride=1, layer_idx=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, rclb_layer=self.rclb_layer, layer_idx=layer_idx, is_ge=self.is_ge))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rclb_layer=self.rclb_layer, layer_idx=layer_idx, is_ge=self.is_ge))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x

def resnet(depth, **kwargs):
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    network = resnet(50, num_classes=10, recalibration_type='srm')
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)
