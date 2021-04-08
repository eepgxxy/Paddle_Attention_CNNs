import paddle.nn as nn
import paddle


class SRMLayer(nn.Layer):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = self.create_parameter(shape=[channel, 2], default_initializer=nn.initializer.Assign(paddle.zeros([channel, 2])))

        self.bn = nn.BatchNorm2D(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.shape

        channel_mean = paddle.mean(paddle.reshape(x, [N, C, -1]), axis=2, keepdim=True)
        channel_var = paddle.var(paddle.reshape(x, [N, C, -1]), axis=2, keepdim=True) + eps
        channel_std = paddle.sqrt(channel_var)

        t = paddle.concat((channel_mean, channel_std), axis=2)
        return t 
    
    def _style_integration(self, t):
        z = t*paddle.reshape(self.cfc, [-1, self.cfc.shape[0], self.cfc.shape[1]])
        tmp = paddle.sum(z, axis=2)
        z = paddle.reshape(tmp, [tmp.shape[0], tmp.shape[1], 1, 1]) # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g

class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.activation = nn.Sigmoid()

        self.reduction = reduction

        self.fc = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(),
                nn.Linear(channel // self.reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        avg_y = paddle.reshape(self.avgpool(x), [b, c])

        gate = paddle.reshape(self.fc(avg_y), [b, c, 1, 1])
        gate = self.activation(gate)

        return x * gate 

class GELayer(nn.Layer):
    def __init__(self, channel, layer_idx):
        super(GELayer, self).__init__()

        # Kernel size w.r.t each layer for global depth-wise convolution
        kernel_size = [-1, 56, 28, 14, 7][layer_idx]

        self.conv = nn.Sequential(
                        nn.Conv2D(channel, channel, kernel_size=kernel_size, groups=channel), 
                        nn.BatchNorm2D(channel),
                    )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        gate = self.conv(x)
        gate = self.activation(gate)

        return x * gate 