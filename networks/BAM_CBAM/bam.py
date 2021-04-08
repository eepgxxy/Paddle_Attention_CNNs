import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Flatten(nn.Layer):
    def forward(self, x):
        return paddle.reshape(x, shape=[x.shape[0], -1])
class ChannelGate(nn.Layer):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_sublayer( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_sublayer( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_sublayer( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1D(gate_channels[i+1]) )
            self.gate_c.add_sublayer( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_sublayer( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.shape[2], stride=in_tensor.shape[2] )
        tmp = paddle.unsqueeze(self.gate_c( avg_pool ), 2)
        tmp = paddle.unsqueeze(tmp, 3)
        result = paddle.expand_as(tmp, in_tensor)
        return result

class SpatialGate(nn.Layer):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_sublayer( 'gate_s_conv_reduce0', nn.Conv2D(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_sublayer( 'gate_s_bn_reduce0',	nn.BatchNorm2D(gate_channel//reduction_ratio) )
        self.gate_s.add_sublayer( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_sublayer( 'gate_s_conv_di_%d'%i, nn.Conv2D(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_sublayer( 'gate_s_bn_di_%d'%i, nn.BatchNorm2D(gate_channel//reduction_ratio) )
            self.gate_s.add_sublayer( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_sublayer( 'gate_s_conv_final', nn.Conv2D(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return paddle.expand_as(self.gate_s( in_tensor ), in_tensor)
class BAM(nn.Layer):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor