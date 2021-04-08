import paddle
import paddle.nn as nn


class sa_layer(nn.Layer):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.cweight = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, channel // (2 * groups), 1, 1])))
        self.cbias = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.ones([1, channel // (2 * groups), 1, 1])))
        self.sweight = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, channel // (2 * groups), 1, 1])))
        self.sbias = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.ones([1, channel // (2 * groups), 1, 1])))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = paddle.reshape(x, [b, groups, -1, h, w])
        x = paddle.transpose(x, [0, 2, 1, 3, 4])

        # flatten
        x = paddle.reshape(x, [b, -1, h, w])

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = paddle.reshape(x, [b * self.groups, -1, h, w])
        x_0, x_1 = paddle.chunk(x, 2, axis=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = paddle.concat([xn, xs], axis=1)
        out = paddle.reshape(out, [b, -1, h, w])

        out = self.channel_shuffle(out, 2)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

class SABottleneck(nn.Layer):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sa = sa_layer(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
            self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, SABottleneck):
                    m.bn3.weight.set_value(paddle.zeros(m.bn3.weight.shape))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x
    

def sa_resnet50(num_classes=1000, pretrained=False):
    print("Constructing sa_resnet50......")
    model = ResNet(SABottleneck, [3, 4, 6, 3], num_classes)
    return model


def sa_resnet101(num_classes=1000, pretrained=False):
    print("Constructing sa_resnet101......")
    model = ResNet(SABottleneck, [3, 4, 23, 3], num_classes)
    return model


def sa_resnet152(num_classes=1000, pretrained=False):
    """Constructs a sa_resnet-152 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SABottleneck, [3, 8, 36, 3], num_classes)
    return model

if __name__ == '__main__':
    network = sa_resnet50(num_classes=10)
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)