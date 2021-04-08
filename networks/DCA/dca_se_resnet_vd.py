#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

__all__ = [
    "DCA_SE_ResNet18_vd", "DCA_SE_ResNet34_vd", "DCA_SE_ResNet50_vd", "DCA_SE_ResNet101_vd",
    "DCA_SE_ResNet152_vd", "DCA_SE_ResNet200_vd"
]

class CSELayer(nn.Layer):
    def __init__(self,in_channel, channel, reduction = 16):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc       = nn.Sequential(
                        nn.Linear(channel, channel // reduction),
                        nn.ReLU(),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
                )
        if in_channel != channel:
            self.att_fc = nn.Sequential(
                nn.Linear(in_channel, channel),
                nn.LayerNorm(channel),
                nn.ReLU()
            )
        self.conv = nn.Sequential(
            nn.Conv2D(2, 1, kernel_size=1),
            nn.LayerNorm(channel),
            nn.ReLU()
        )


    def forward(self, x):
        b, c, _, _ = x[0].shape
        gap = paddle.reshape(self.avg_pool(x[0]), [b, c])
        if x[1] is None:
            all_att = self.fc(gap)
        else:
            pre_att = self.att_fc(x[1]) if hasattr(self, 'att_fc') else x[1]
            all_att = paddle.concat([paddle.reshape(gap, (b, 1, 1, c)), paddle.reshape(pre_att, (b, 1, 1, c))], axis=1)
            all_att = paddle.reshape(self.conv(all_att), (b, c))
            all_att = self.fc(all_att)
        return {0: x[0] * paddle.reshape(all_att, (b, c, 1, 1)), 1: gap*all_att}

class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 reduction_ratio=16,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")
        self.se = CSELayer(num_channels, num_filters * 4)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs[0])
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        out = self.se({0:conv2, 1:inputs[1]})

        if self.shortcut:
            short = inputs[0]
        else:
            short = self.short(inputs[0])
        out_x = paddle.add(x=short, y=out[0])
        out_x = F.relu(out_x)
        out_att = out[1]
        return {0: out_x, 1: out_att}


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 reduction_ratio=16,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")

        self.se  = CSELayer(num_channels,num_filters)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs[0])
        conv1 = self.conv1(y)
        out = self.se({0:conv1, 1:inputs[1]})

        if self.shortcut:
            short = inputs[0]
        else:
            short = self.short(inputs[0])
        out_x = paddle.add(x=short, y=out[0])
        out_x = F.relu(out_x)
        out_att = out[1]
        return {0: out_x, 1: out_att}


class DCA_SE_ResNet_vd(nn.Layer):
    def __init__(self, layers=50, class_dim=1000):
        super(DCA_SE_ResNet_vd, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=32,
            filter_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            num_channels=32,
            num_filters=32,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            num_channels=32,
            num_filters=64,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    self.block_list.append(basic_block)
                    shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc6_weights"),
            bias_attr=ParamAttr(name="fc6_offset"))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        att = None
        y = {0: y, 1: att}
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y[0])
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def DCA_SE_ResNet18_vd(**args):
    model = DCA_SE_ResNet_vd(layers=18, **args)
    return model


def DCA_SE_ResNet34_vd(**args):
    model = DCA_SE_ResNet_vd(layers=34, **args)
    return model


def DCA_SE_ResNet50_vd(**args):
    model = DCA_SE_ResNet_vd(layers=50, **args)
    return model


def DCA_SE_ResNet101_vd(**args):
    model = DCA_SE_ResNet_vd(layers=101, **args)
    return model


def DCA_SE_ResNet152_vd(**args):
    model = DCA_SE_ResNet_vd(layers=152, **args)
    return model


def DCA_SE_ResNet200_vd(**args):
    model = DCA_SE_ResNet_vd(layers=200, **args)
    return model

def demo():
    net = DCA_SE_ResNet50_vd(class_dim=10)
    y = net(paddle.randn([2, 3, 224,224]))
    print(y.shape)

# demo()
