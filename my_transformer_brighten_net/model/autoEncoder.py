
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import os
import math
# from timm.models.layers import trunc_normal_

# from model.blocks import CBlock_ln, SwinTransformerBlock
# from model.global_net import Global_pred

from blocks import CBlock_ln, SwinTransformerBlock
from global_net import Global_pred

# 构建网络
# 2*autoencoder + 简陋gamma网络 
class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder1
        self.Gamma = nn.Sequential(
            nn.Conv2d(3, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(285180, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.Encoder1 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(3, 64, 3, 1, 1),   # [, 64, 96, 96]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), # [, 64, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1), # [, 256, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # [, 256, 24, 24]
            nn.BatchNorm2d(256)   
        )
        
        # decoder1
        self.Decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),   # [, 128, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 96, 96]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 96, 96]
            nn.Sigmoid()
        )

         # Encoder2
        self.Encoder2 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(3, 64, 3, 1, 1),   # [, 64, 96, 96]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), # [, 64, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1), # [, 256, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # [, 256, 24, 24]
            nn.BatchNorm2d(256)   
        )
        
        # decoder2
        self.Decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),   # [, 128, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 96, 96]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 96, 96]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoder1 = self.Encoder1(x)
        decoder1 = self.Decoder1(encoder1)
        encoder2 = self.Encoder2(x)
        decoder2 = self.Decoder2(encoder2)
        gamma = self.Gamma(x)

        img_mid = (x*decoder1)+decoder2
        b = img_mid.shape[0]
        img_high = img_mid**gamma if b==1 else torch.stack([img_mid[i,:,:,:]**gamma[i,:] for i in range(b)], dim=0)
        return img_high
        


class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
            
            

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)

        return mul, add



# 2*local网络 + 简陋gamma网络
class ModelV1(nn.Module):
    def __init__(self, in_dim=3):
        super(ModelV1, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.global_net = Global_pred(in_channels=in_dim, type='lol')
        # self.Gamma = nn.Sequential(
        #     nn.Conv2d(3, 10, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(10, 20, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(285180, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 1),
        #     nn.Sigmoid()
        # )
    def forward(self,img_low):
        mul, add = self.local_net(img_low)
        gamma, color = self.global_net(img_low)
        img_mid = (img_low*mul)+add
        b = img_mid.shape[0]
        img_high = img_mid**gamma if b==1 else torch.stack([img_mid[i,:,:,:]**gamma[i,:] for i in range(b)], dim=0)
        return img_high


# 2*local网络 + Transformer gamma网络
# class ModelV2(nn.Module):


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='3'
    img = torch.Tensor(1, 3, 400, 600)
    DAEmodel = DenoiseAutoEncoder()
    # DAEmodel = ModelV1()
    print('total parameters:', sum(param.numel() for param in DAEmodel.parameters()))
    a = DAEmodel(img)
    print('res',a.shape)
    torch.save(DAEmodel.state_dict(), 'model_v1.pt')

    #查看参数量和计算量
    import torch
    from thop import profile
    input = torch.randn(1, 3, 400, 600)
    flops, params = profile(DAEmodel, inputs=(input, ))
    print("flops:", flops,"\nparams:", params)


