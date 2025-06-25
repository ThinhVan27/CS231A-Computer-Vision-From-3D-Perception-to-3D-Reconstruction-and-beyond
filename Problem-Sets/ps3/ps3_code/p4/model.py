import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class DispLayer(nn.Module):
    def __init__(self, in_channels, interm_channels):
        super(DispLayer, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, interm_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(interm_channels)
        self.conv2 = nn.Conv2d(interm_channels, 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.bn2(self.conv2(x)))
        return 0.3 * x


class UpLayer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(UpLayer, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, output_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(self.upsample1(x))))


class ResNetDisparity(resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetDisparity, self).__init__(*args, **kwargs)
        inplanes = (64 * self.layer1[-1].expansion,
                    128 * self.layer1[-1].expansion,
                    256 * self.layer1[-1].expansion,
                    512 * self.layer1[-1].expansion)

        self.up1 = UpLayer(inplanes[0] * 2, inplanes[0])
        self.up2 = UpLayer(inplanes[1] * 2, inplanes[0])
        self.up3 = UpLayer(inplanes[2] * 2, inplanes[1])
        self.up4 = UpLayer(inplanes[3], inplanes[2])

        self.disp1 = DispLayer(inplanes[0], 32)
        self.disp2 = DispLayer(inplanes[0], 32)
        self.disp3 = DispLayer(inplanes[1], 32)
        self.disp4 = DispLayer(inplanes[2], 32)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.up4(x4)
        disp4 = self.disp4(x4)

        x3 = torch.cat((x3, x4), dim=1)
        x3 = self.up3(x3)
        disp3 = self.disp3(x3)

        x2 = torch.cat((x2, x3), dim=1)
        x2 = self.up2(x2)
        disp2 = self.disp2(x2)

        x1 = torch.cat((x1, x2), dim=1)
        disp1 = self.disp1(self.up1(x1))

        return [disp1, disp2, disp3, disp4]


def _resnet_disp(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetDisparity(block, layers, **kwargs)
    if pretrained:
        state_dict = resnet.load_state_dict_from_url(resnet.model_urls[arch],
                                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnetdisp18(pretrained=False, progress=True, **kwargs):
    return _resnet_disp('resnet18', resnet.BasicBlock, [2, 2, 2, 2],
                        pretrained, progress, **kwargs)


def resnetdisp34(pretrained=False, progress=True, **kwargs):
    return _resnet_disp('resnet34', resnet.BasicBlock, [3, 4, 6, 3],
                        pretrained, progress, **kwargs)


def resnetdisp50(pretrained=False, progress=True, **kwargs):
    return _resnet_disp('resnet50', resnet.Bottleneck, [3, 4, 6, 3],
                        pretrained, progress, **kwargs)


def resnetdisp101(pretrained=False, progress=True, **kwargs):
    return _resnet_disp('resnet101', resnet.Bottleneck, [3, 4, 23, 3],
                        pretrained, progress, **kwargs)
