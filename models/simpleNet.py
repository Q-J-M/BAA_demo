import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling_attention(nn.Module):
    def __init__(self, input_channels, kernel_size=1):
        super(Pooling_attention, self).__init__()
        self.pooling_attention = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.pooling_attention(x)


class Part_Relation(nn.Module):
    def __init__(self, input_channels, reduction=[2, 4], level=2):
        super(Part_Relation, self).__init__()

        modules = []
        for i in range(level):
            output_channels = input_channels // reduction[i]
            modules.append(nn.Conv2d(input_channels, output_channels, kernel_size=1))
            modules.append(nn.BatchNorm2d(output_channels))
            modules.append(nn.ReLU())
            input_channels = output_channels

        self.pooling_attention_0 = nn.Sequential(*modules)
        self.pooling_attention_1 = Pooling_attention(input_channels, 1)
        self.pooling_attention_3 = Pooling_attention(input_channels, 3)
        self.pooling_attention_5 = Pooling_attention(input_channels, 5)

        self.last_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.pooling_attention_0(x)
        x = torch.cat([self.pooling_attention_1(x), self.pooling_attention_3(x), self.pooling_attention_5(x)], dim=1)
        x = self.last_conv(x)
        return input - input * x


class SimpleNet(nn.Module):
    def __init__(self, gender_encode_length):
        super(SimpleNet, self).__init__()


        self.backbone0 = nn.Sequential(
            nn.Conv2d(10, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.part_relation0 = Part_Relation(128)


        self.backbone1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.part_relation1 = Part_Relation(256)


        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)


        self.fc0 = nn.Linear(256 + gender_encode_length, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, 1)

    def forward(self, image, gender):

        x = self.part_relation0(self.backbone0(image))
        x = self.part_relation1(self.backbone1(x))


        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, 256)
        # image_feature = x


        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)


        x = torch.cat([x, gender_encode], dim=1)


        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.output(x)

        return x
