import paddle
from paddle import nn


class eca_layer(nn.Layer):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.shape

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = paddle.squeeze(y, axis=-1)
        y = paddle.transpose(y, perm=[0, 2, 1])
        y = self.conv(y)
        y = paddle.transpose(y, perm=[0, 2, 1])
        paddle.unsqueeze_(y, axis=-1)
        
        # Multi-scale information fusion
        y = self.sigmoid(y)
        y = paddle.expand_as(y, x)

        return x * y


if __name__ == '__main__':
    network = eca_layer(5)
    img = paddle.zeros([8, 3, 224, 224])
    outs = network(img)
    print(outs.shape) #[8, 3, 224, 224]