import torch
import torch.nn as nn




# RLA channel k: rla_channel = 32 (default)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#=========================== define bottleneck ============================
class IIM_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 rla_channel=64, SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(IIM_BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Define conv-bn-relu for the first part of the block
        self.conv1 = conv3x3(inplanes + rla_channel, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_=conv3x3(planes,planes)
        self.bn_ = norm_layer(planes)
        self.relu_ = nn.ReLU(inplace=True)
        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

        # SE and ECA blocks - optional
        self.se = None
        if SE:
            self.se = SELayer(planes * self.expansion, reduction)

        self.eca = None
        if ECA_size is not None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))

    def forward(self, x, h):
        identity = x

        # Apply conv-bn-relu operations to the input
        x = torch.cat((x, h), dim=1)  # Concatenate the input and skip connection

        out = self.conv1(x)  # [batch, planes, height, width]
        out = self.bn1(out)  # [batch, planes, height, width]
        out = self.relu(out)

        out = self.conv2(out)  # [batch, planes, height, width]
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.eca is not None:
            out = self.eca(out)

        y = out

        # Apply conv-bn-relu operations to the identity (skip connection)
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)
        # Ensure the skip connection is processed with conv-bn-relu
        identity = self.conv_(identity)  # Apply same conv-bn-relu to identity
        identity = self.bn_(identity)
        identity = self.relu_(identity)

        # Add the processed skip connection to the output
        out += identity
        out = self.relu(out)
        return out, y, h



class IIM_ResNet34(nn.Module):

    def __init__(self, block, layers, num_classes=1000, 
                 rla_channel=64, SE=True, ECA=None,
                 zero_init_last_bn=True, #zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(IIM_ResNet34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if ECA is None:
            ECA = [None] * 4
        elif len(ECA) != 4:
            raise ValueError("argument ECA should be a 4-element tuple, got {}".format(ECA))
        
        self.rla_channel = rla_channel
        self.flops = False
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        conv_outs = [None] * 4
        # recurrent_convs = [None] * 4
        stages = [None] * 4
        stage_bns = [None] * 4
        
        stages[0], stage_bns[0], conv_outs[0] = self._make_layer(block, 64, layers[0], 
                                                                                     rla_channel=rla_channel, SE=SE, ECA_size=ECA[0])
        stages[1], stage_bns[1], conv_outs[1] = self._make_layer(block, 128, layers[1], 
                                                                                     rla_channel=rla_channel, SE=SE, ECA_size=ECA[1], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[0])
        stages[2], stage_bns[2], conv_outs[2] = self._make_layer(block, 256, layers[2], 
                                                                                     rla_channel=rla_channel, SE=SE, ECA_size=ECA[2], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[1])
        stages[3], stage_bns[3], conv_outs[3] = self._make_layer(block, 512, layers[3],
                                                                                     rla_channel=rla_channel, SE=SE, ECA_size=ECA[3], 
                                                                                     stride=2, dilate=replace_stride_with_dilation[2])
        
        self.conv_outs = nn.ModuleList(conv_outs)
        # self.recurrent_convs = nn.ModuleList(recurrent_convs)
        self.stages = nn.ModuleList(stages)
        self.stage_bns = nn.ModuleList(stage_bns)
        
        self.tanh = nn.Tanh()
        
        self.bn2 = norm_layer(rla_channel)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion) + rla_channel, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, 
                    rla_channel, SE, ECA_size, stride=1, dilate=False):
        
        conv_out = conv1x1(int(planes * block.expansion), rla_channel)
        # recurrent_conv = conv3x3(rla_channel, rla_channel)
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != int(planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, int(planes * block.expansion), stride),
                norm_layer(int(planes * block.expansion)),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            rla_channel=rla_channel, SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = int(planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                rla_channel=rla_channel, SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        bns = [norm_layer(64) for _ in range(blocks)]

        return nn.ModuleList(layers), nn.ModuleList(bns), conv_out

    
    def _get_one_layer(self, layers, bns, conv_out, x, h):
        for layer, bn in zip(layers, bns):
            x, y, h = layer(x, h)
            y_out = conv_out(y)
            h = h + y_out
            h = bn(h)
            h = self.tanh(h)
    
        return x, h
        

    def _forward_impl(self, x):
        x = self.conv1(x)     # [4, 3, 224, 224], [4, 64, 112, 112]
        x = self.bn1(x)       # [4, 64, 112, 112]
        x = self.relu(x)
        x = self.maxpool(x)   # [4, 64, 56, 56]

        batch, _, height, width = x.size()   # [4, 64, 56, 56]
        # self.rla_channel = rla_channel

        h = torch.zeros(batch, self.rla_channel, height, width, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # h = torch.zeros(batch, self.rla_channel, height, width)   # [4, 32, 56, 56]

        # layer_0
        layer_0 = self.stages[0]
        bns_0 = self.stage_bns[0]
        conv_out_0 = self.conv_outs[0]
        # recurrent_conv_0 = self.recurrent_convs[0]
        x_1, h_1 = self._get_one_layer(layer_0, bns_0, conv_out_0, x, h)    # # output x_1  # [8, 256, 56, 56], [8, 32, 56, 56]
        
                
        # layer_1
        layer_1 = self.stages[1]
        bns_1 = self.stage_bns[1]
        conv_out_1 = self.conv_outs[1]
        # recurrent_conv_1 = self.recurrent_convs[1]
        x_2, h_2 = self._get_one_layer(layer_1, bns_1, conv_out_1, x_1, h_1)    # output x_2 [8, 512, 28, 28], [8, 32, 28, 28]
        
        
        # layer_2
        layer_2 = self.stages[2]
        bns_2 = self.stage_bns[2]
        conv_out_2 = self.conv_outs[2]
        # recurrent_conv_2 = self.recurrent_convs[2]
        x_3, h_3 = self._get_one_layer(layer_2, bns_2, conv_out_2, x_2, h_2)    # output x_3 [8, 1024, 14, 14], [8, 32, 14, 14]
        

        # layer_3
        layer_3 = self.stages[3]
        bns_3 = self.stage_bns[3]
        conv_out_3 = self.conv_outs[3]
        # recurrent_conv_3 = self.recurrent_convs[3]
        x_4, h_4 = self._get_one_layer(layer_3, bns_3, conv_out_3, x_3, h_3)    # output x_4 [8, 2048, 7, 7], [8, 32, 7, 7]
        
        return x_1, x_2, x_3, x_4, h_1, h_2, h_3, h_4

       
    def forward(self, x):
        return self._forward_impl(x)




    


def IIM_resnet34(rla_channel=64, k_size=[5, 5, 5, 7]):
    print("Constructing IIM_ResNet34......")
    model = IIM_ResNet34(IIM_BasicBlock, [3, 4, 6, 3], rla_channel=rla_channel, ECA=k_size, SE=False)
    return model




