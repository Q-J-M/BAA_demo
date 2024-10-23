import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D_Base(nn.Module):
    def __init__(self, input_channels, output_channels,dropout_prob=0.2):
        super(UNet3D_Base, self).__init__()

        self.dropout_prob = dropout_prob

        # encode
        self.enc1 = self.contract_block(input_channels, 8)
        self.enc2 = self.contract_block(8, 16)
        self.enc3 = self.contract_block(16, 32)
        self.enc4 = self.contract_block(32, 64)

        # decode
        self.dec1 = self.expand_block(64, 32)
        self.dec2 = self.expand_block(32, 16)
        self.dec3 = self.expand_block(16, 8)
        self.dec4 = nn.Conv3d(8, output_channels, kernel_size=1)

        self.conv_adjust_enc3 = nn.Conv3d(64, 32, kernel_size=1)
        self.conv_adjust_enc2 = nn.Conv3d(32, 16, kernel_size=1)
        self.conv_adjust_enc1 = nn.Conv3d(16, 8, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=self.dropout_prob)
        )

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.dropout_prob)
        )

    def forward(self, x):
        # encode
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # decode
        dec1 = self.dec1(enc4)
        dec1 = self.conv_adjust_enc3(torch.cat((dec1, enc3), dim=1))
        dec2 = self.dec2(dec1)
        dec2 = self.conv_adjust_enc2(torch.cat((dec2, enc2), dim=1))
        dec3 = self.dec3(dec2)
        dec3 = self.conv_adjust_enc1(torch.cat((dec3, enc1), dim=1))
        dec4 = self.dec4(dec3)

        return dec4



    # def forward(self, x):
    #
    #     enc1 = self.enc1(x)
    #     enc2 = self.enc2(enc1)
    #     enc3 = self.enc3(enc2)
    #
    #
    #     dec1 = self.dec2(enc3)
    #     dec1 = torch.cat((dec1, enc2), dim=1)
    #     dec2 = self.dec3(dec1)
    #     dec2 = torch.cat((dec2, enc1), dim=1)
    #     dec3 = self.dec4(dec2)
    #
    #     return dec3


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim):
#         super(MLP, self).__init__()
#         layers = []
#
#
#         layers.append(nn.Linear(input_dim, hidden_dims[0]))
#         layers.append(nn.ReLU())
#
#
#         for i in range(1, len(hidden_dims)):
#             layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
#             layers.append(nn.ReLU())
#
#
#         layers.append(nn.Linear(hidden_dims[-1], output_dim))
#
#
#         self.mlp = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.mlp(x)



# class UNet3D(nn.Module):
#     def __init__(self, input_channels, output_channels, gender_encode_length):
#         super(UNet3D, self).__init__()
#         self.unet = UNet3D_Base(input_channels, output_channels)
#         self.output_channels = output_channels
#
#         self.gender_encoder = nn.Linear(1, gender_encode_length)
#         self.gender_bn = nn.BatchNorm1d(gender_encode_length)
#
#         self.fc0 = nn.Linear(output_channels + gender_encode_length, 1024)
#         self.fc1 = nn.Linear(1024, 512)
#         self.output = nn.Linear(512, 1)
#
#         #
#         # input_dim = 256
#         # hidden_dims = [512, 256, 128]
#         # output_dim = 1
#         # self.mlp = MLP(input_dim, hidden_dims, output_dim)
#
#     def forward(self, x, gender):
#
#         output = self.unet(x)
#         output = F.adaptive_avg_pool3d(output, (1, 1, 1))
#         output = output.view(self.output_channels, -1)
#
#
#         gender_encode = self.gender_bn(self.gender_encoder(gender))
#         gender_encode = F.relu(gender_encode)
#
#         x = torch.cat([output, gender_encode], dim=1)
#         x = F.relu(self.fc0(x))
#
#         x = F.relu(self.fc1(x))
#
#         mlp_output = self.output(x)
#
#
#         # output = output + gender_encode.view(-1, 1, 1, gender_encode.size(1))
#
#         # mlp_output = self.mlp(output)
#
#         return mlp_output

class UNet3D(nn.Module):
    def __init__(self, input_channels, output_channels, gender_encode_length):
        super(UNet3D, self).__init__()
        self.unet = UNet3D_Base(input_channels, output_channels,dropout_prob=0.2)
        self.output_channels = output_channels

        self.gender_encoder = nn.Linear(1, gender_encode_length)
        self.gender_bn = nn.BatchNorm1d(gender_encode_length)


        self.flattened_dim = 8*8*16 + gender_encode_length

        self.fc0 = nn.Linear(self.flattened_dim, 512)
        self.dropout0 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(p=0.2)

        self.output = nn.Linear(256, 1)

    def forward(self, x, gender):

        output = self.unet(x)


        output = F.adaptive_avg_pool3d(output, (8, 8,16))
        output = output.view(output.size(0), -1)

        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)


        x = torch.cat([output, gender_encode], dim=1)

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))

        mlp_output = self.output(x)

        return mlp_output
