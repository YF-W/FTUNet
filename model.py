import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as resnet_model
import torchvision.transforms.functional as TF
import numpy as np

"""
FTUNet

Author: Yuefei Wang
Affiliation: Chengdu University
For academic and research purposes only, not for commercial use.

"""

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=16):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(outplanes * 5, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class ASPPInception(nn.Module):
    def __init__(self, in_channel, ASPP_outc, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(ASPPInception, self).__init__()

        self.aspp = ASPP(inplanes=in_channel, outplanes=ASPP_outc)

        # 定义inception模块第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # 定义inception模块的第三条线路
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # 定义inception模块第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

        self.ConvtoReduceDimention = nn.Conv2d(ASPP_outc * 4, ASPP_outc, kernel_size=1)

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        x = self.aspp(imgs)

        b2 = self.branch3x3(imgs)
        b3 = self.branch5x5(imgs)
        b4 = self.branch_pool(imgs)

        ConcatResult = torch.cat((x, b2, b3, b4), dim=1)

        output = self.ConvtoReduceDimention(ConcatResult)

        return output

class MS(nn.Module):
    def __init__(self):
        super(MS,self).__init__()
    def forward(self,x,):
        b,c,h=x.size()
        y=x.detach().cpu().numpy()
        for i in range(b):
            mid=np.median(y[i])
            for j in range(c):
                for k in range(h):
                    y[i][j][k] = (y[i][j][k] - mid) ** 3
        x=torch.tensor(y)
        x=x.cuda(0)
        return x

class PA(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(PA, self).__init__()

        # Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # CFC: channel-wise fully connected layer
        self.mb=MS()
        self.cfc = nn.Conv1d(channel, channel, kernel_size=3, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        # mean = x.view(b, c, -1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        mean2 = self.avg_pool(x).squeeze(-1)
        max = self.max_pool(x).squeeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean2, max, std), -1)  # (b, c, 2)
        # u = torch.cat((mean, max, std), -1)  # (b, c, 2)

        # z=CFC(b,c)
        # Style integration
        u=self.mb(u)
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class FTUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FTUNet, self).__init__()

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)  # 自动下载预训练模型文件用于用以加载训练模型

        resnet = resnet_model.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        features=[3,64,128,256,512]
        self.srm1 = PA(256)
        self.srm2 = PA(512)
        self.srm3 = PA(1024)
        self.srm4 = PA(2048)

        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=2048, kernel_size=1, padding=0)
        self.conv_seq_img1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1, padding=0)
        self.conv_seq_img2 = nn.Conv2d(in_channels=12, out_channels=512, kernel_size=1, padding=0)
        self.conv_seq_img3 = nn.Conv2d(in_channels=48, out_channels=1024, kernel_size=1, padding=0)


        self.conv_cat_res_trans1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.conv_cat_res_trans2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        self.conv_cat_res_trans3 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, padding=0)

        self.relu=nn.ReLU()
        self.bn1_att=nn.BatchNorm2d(512)
        self.bn2_att=nn.BatchNorm2d(1024)
        self.bn3_att=nn.BatchNorm2d(2048)

        layer_in_channel = 512
        layer_out_channels = 1024
        self.aspp_inception = ASPPInception(in_channel=layer_in_channel, ASPP_outc=layer_out_channels,
                                            out2_1=int(layer_out_channels / 2), out2_3=layer_out_channels,
                                            out3_1=int(layer_out_channels / 2), out3_5=layer_out_channels,
                                            out4_1=layer_out_channels)
        self.conv1 = nn.Conv2d(in_channels=4096, out_channels=512, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.decoder4 = DecoderBottleneckLayer(512, 256)
        self.decoder3 = DecoderBottleneckLayer(1280, 128)
        self.decoder2 = DecoderBottleneckLayer(640, 64)
        self.decoder1 = DecoderBottleneckLayer(320, 64)

        self.final_conv1 = nn.ConvTranspose2d(320, 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        skip_connections = []
        b, c, h, w = x.shape
        y=x
        z=x

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        emb = self.patch_embed(x)
        feature_tf_list=[]
        for i in range(12):
            emb = self.transformers[i](emb)
            if(i%4==0):
                feature_tf_list.append(emb)
        skip_connections.append(e1)
        skip_connections.append(e2)
        skip_connections.append(e3)
        skip_connections.append(e4)

        a1 = self.srm1(skip_connections[0])
        a2 = self.srm2(skip_connections[1])
        a3 = self.srm3(skip_connections[2])

        feature_tf_list[0] = feature_tf_list[0].permute(0, 2, 1)
        feature_tf_list[0] = feature_tf_list[0].view(b, 3, 112, 112)
        feature_tf_list[0] = self.conv_seq_img1(feature_tf_list[0])

        d1=e1
        e1 = torch.cat((a1, feature_tf_list[0]), dim=1)
        e1=self.bn1_att(e1)
        e1=self.relu(e1)
        e1=self.conv_cat_res_trans1(e1)
        e1=e1+d1

        feature_tf_list[1] = feature_tf_list[1].permute(0, 2, 1)
        feature_tf_list[1] = feature_tf_list[1].view(b, 12, 56, 56)
        feature_tf_list[1] = self.conv_seq_img2(feature_tf_list[1])

        d2=e2
        e2 = torch.cat((a2, feature_tf_list[1]), dim=1)  # 640
        e2=self.bn2_att(e2)
        e2=self.relu(e2)
        e2 = self.conv_cat_res_trans2(e2)
        e2=e2+d2

        feature_tf_list[2] = feature_tf_list[2].permute(0, 2, 1)
        feature_tf_list[2] = feature_tf_list[2].view(b, 48, 28, 28)
        feature_tf_list[2] = self.conv_seq_img3(feature_tf_list[2])

        d3=e3
        e3 = torch.cat((a3, feature_tf_list[2]), dim=1)
        e3=self.bn3_att(e3)
        e3=self.relu(e3)
        e3 = self.conv_cat_res_trans3(e3)
        e3=e3+d3

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((e4, feature_tf), dim=1)
        feature_att = self.conv1(feature_cat)
        # feature_att=self.bottleneck(feature_att)
        feature_att = self.aspp_inception(feature_att)
        feature_out = self.conv2(feature_att)


        d4 = self.decoder4(feature_out)
        d4=torch.cat([d4,e3],dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out
