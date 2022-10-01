import torch
import torch.nn as nn
from ..utils import load_state_dict_from_url
# reference: https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
# reference: https://yhkim4504.tistory.com/3
# * qwe
# ! qwe
# ? qwe
# TODO qwe

# flow: conv1x1, 3x3 / BasicBlock / Bottleneck / ResNet(init, make_layer, forward) / resnets / notes

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
# !
# in_planes: in_channels(입력 필터개수)
# out_planes: out_channels(출력 필터개수)
# groups: input과 output의 connection을 제어, default=1
# dilation: 커널 원소간의 거리. 늘릴수록 같은 파라미터수로 더 넓은 범위를 파악할 수 있음
#           default=1.
# bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    # !
    # inplanes: input channel size
    # planes: output channel size
    # groups, base_width: ResNeXt, Wide ResNet의 경우 사용.
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d #* default norm_layer --> BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # The structure of Basic Block
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes) # ! bn: Batch Normalization // >>변형된 분포가 나오지 않도록 하기 위해.
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # * x

        out = self.conv1(x) # * from here
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) # * to here, F(x)

        if self.downsample is not None:
            # x와 F(x)의 tensor size가 다를 때,
            identity = self.downsample(x)
        # * identity mapping시, mapping후 ReLU를 적용한다.
        # * ReLU를 통과하면 양의 값만 남아 Residual의 의미가 제대로 유지되지 않기 때문.
        out += identity #* F(x)+x
        out = self.relu(out) #* activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # ! Block내의 마지막 conv1x1에서 차원 증사시키는 확장 계수

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # * ResNeXt, WideResNet의 경우 사용
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # * The structure of Bottleneck Block
        self.conv1 = conv1x1(inplanes, width) # 논문 실험들에선 여기서 차원 축소 -> 3x3에서 연산부담 줄임
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # ! conv2에서 downsample, stride=2.
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 다시 차원 증가
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # conv
        out = self.bn1(out) # batch norm
        out = self.relu(out)# ReLU
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) # 여기까지 F(x)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity 
        out = self.relu(out) # * H(x) = ReLU(F(x) + x)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # * default values
        self.inplanes = 64 # input feature map
        self.dilation = 1
        # * stride를 dilation으로 대체할지 선택.
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # * 7x7 conv, 3x3 max pooling
        # * 3: input이 RGB이미지여서 conv layer의 input channel수는 3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ### * residual blocks
        # * filter의 개수는 각 block들을 거치면서 증가. 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # * layer --end--
        
        # * 모든 block을 거친 후엔 AvgPolling으로 (n, 512, 1, 1)의 텐서로 변환
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # * fully connected layer에 연결
        # 이전 레이어의 출력을 평탄화하여 다음 stage의 입력이 될 수 있는 단일 벡터로 변환한다.
        # falt -> activation -> Softmax
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # !
        # * 논문의 conv2_x, conv3_x, ... 5_x를 구현하며,
        # * 각 층에 해당하는 block을 개수에 맞게 생성 및 연결시켜주는 역할을 한다.
        # * convolution layer 생성 함수
        # block: block 종류
        # planes: feature map size (input shape)
        # blocks: layers[0], [1], ... 과 같이, 해당 블록이 몇개 생성돼야 하는지, 블록의 개수 (layer 반복해서 쌓는 개수)
        # stride, dilate is fixed.
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # stride가 1이 아님-->크기가 줄어듦, 차원의 크기가 맞지 않음 --> downsampling
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 블록 내 시작 layer, downsampling해야함.
        # 왜 처음 한번은 따로 쌓음? -> 첫 block을 쌓고 self.inplanes을 plane에 맞춰주기 위함.
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion # * update inplanes
        # 동일 블록 반복.
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # !
    # arch: ResNet model 이름
    # block: 어떤 block형태를 사용할지 (Basic or Bottleneck)
    # layers: 해당 block이 몇번 사용되는지를 list형태로 넘겨주는 부분
    # pretrained: pretrain된 model weights를 불러오기
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

# !
# 공통 Arguments
## pretrained (bool): If True, returns a model pre-trained on ImageNet
## progress (bool): If True, displays a progress bar of the download to stderr
def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32 # input channel을 32개의 그룹으로 분할(cardinality)
    kwargs['width_per_group'] = 4 # 각 그룹당 4(=128/32)개의 채널로 구성
    # 각 그룹당 channel=4의 output feature map 생성, concatenate하여 128개로 다시 생성.
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2 #* base_width = 64*2로 증가
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
    
    
'''
[주요 개념]
1. skip connection:
    ResNet에선 두 가지 형태의 skip connection을 다룬다.
    (1) Identity Shortcut: F(x)+x의 구조이며, element-wise addition에 학습대상 parameter가 없다.
    (2) Projection Shortcut: F(x) + Wx(W: convolution)의 구조이며, F(x)와 x의 차원이 다를 때 사용한다.
    1x1 conv + bn의 구조를 이요해 차원을 맞춰주며, conv를 사용하므로 학습대상 param이 존재한다.
    
2. down sampling:
    더 작은 이미지로 크기를 축소시키는 것.
    VGG net에선 output feature map크기를 줄일 때 max pooling을 사용했지만,
    ResNet은 복잡도를 줄이기 위해 stride=2로 대체했다.
    
    ResNet에선 차원이 바뀌는 블록의 첫 번째 conv layer에서 stride=2로 사용하여 feature map크기를 줄였다.
    즉, conv3 1, conv4 1, conv 5 1에서 사용된다.
    
3. Block(Resudual Block):
    ResNet에선 34-layer까지는 Basic Block을 사용하고, 더 깊은 구조엔 Bottleneck구조를 사용한다.
    (깊이++ -> param 수가 많아짐 -> 50층 이상에선 Bottleneck사용)
    
    -Basic Block: 3x3 conv + 3x3 conv구조
    
    -Bottleneck: 1x1 conv -> 3x3 conv -> 1x1 conv구조
    처음 1x1 conv에서 차원 축소 => 3x3 conv에선 연산부담-- => 마지막 1x1 conv에서 다시 차원 증가
    # ! 실제 코드에선 3x3에서 차원 축소(down sampling을 함) why?
    In all experiments in the paper, the stride=2 operation is in the first 1x1 conv layer when downsampling. 
    This might not be the best choice, as it wastes some computations of the preceding block. 
    For example, using stride=2 in the first 1x1 conv in the first block of conv3 is 
    equivalent to using stride=2 in the 3x3 conv in the last block of conv2. 
    So I feel applying stride=2 to either the first 1x1 or the 3x3 conv should work. 
    I just kept it “as is”, because we do not have enough resources to investigate every choice.
    # ! reference: https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/cyjqnkv/
    
[argument]
1. dilation:
    kernel원소 사이의 간격(default=1). 
    dilation rate에 맞춰 filter 칸들 사이에 zero padding을 해 크기를 늘려준다.
    ex) 3x3 filter면, filter원소가 1이라 가정.
            10101
    111     00000
    111  -> 10101
    111     00000
            10101 같은 형태임.
        
    * 필터 사이즈가 커지면서 시야가 넓어진다는 장점. 
    * 주로 real-time segmentation에서 효과가 있다고 알려짐.
    
    ResNet에서 input size와 output size가 동일해야 할 때 padding=dilation으로 사용.
    3x3 filter with dilation=2 -> 5x5와 동일한 크기가 됨. -> padding도 2로 해야 output size가 맞게 됨.
    
2. width, group
    WideResNet, ResNeXt구현시 사용됨.
    
    -WideResNet의 경우:
      width_per_group을 사용한다. in general WRN_n_k로 표기.
      n is total number of layer(depth),
      k is widening factors(width, 폭).
      
      k=1일때, ResNet과 동일한 width인 경우이며, k가 1보다 큰 만큼 일반 ResNet보다 넓이가 k배 넓게 구현된다.
      ex) wide_resnet50_2 model은 총 50층, widening 계수(k)는 2인 모델이며,
      kwargs['width_per_group'] = 64*2로 base_width=64보다 폭이 2배로 증가된다.
      
      
    -ResNeXt의 경우:
      groups와 width_per_group을 사용해 Cardinality개념을 적용한다.
      ex) resnet50_32x4d model은 kwargs['group']=32로 input channel을 32개 그룹으로 분할(Cardinality)하고,
      kwargs['width_per_group']=4로 각 그룹당 4(=128/32)개의 채널로 구성한다.
'''

''' sample code
model = ResNet(BasicBlock, [2,2,2,2])
x = torch.randn(1, 3, 112, 112)
print('\noutput shape: ', model(x).shape)
'''