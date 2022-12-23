#将pth 转为coreML
#torch version 1.12.1



import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg # mpimg 用于读取图片
# import pandas as pd 
# from sklearn.model_selection import train_test_split
# from skimage.util import random_noise
# from skimage.metrics import peak_signal_noise_ratio
import torch
from torch import nn
# import torch.nn.functional as F
# import torch.utils.data as Data 
# import torch.optim as optim
# from torchvision import transforms
# from torchvision.datasets import STL10
# import hiddenlayer as hl
# from tqdm import tqdm
# import random
# import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
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
        
        # decoder
        self.Decoder = nn.Sequential(
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
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder
        
# 输出网络结构
DAEmodel = DenoiseAutoEncoder().to(device)
DAEmodel.load_state_dict(torch.load('autodecode.mdl'))
DAEmodel.eval()


#直接用coremltools，会有问题，输出结构无法设置。
# import coremltools as ct
# random_input = torch.rand(1, 3, 96, 96) 
# traced_model = torch.jit.trace(DAEmodel, random_input) 
# model = ct.convert(
#     traced_model,
#     inputs=[ct.ImageType(name="input_1", shape=random_input.shape)],
# ) 
# model.save("imgDenoise.mlmodel")


##尝试使用 pth => onnx => mlmodel
#转换为onnx
dummy_input = torch.rand(1, 3, 96, 96) #这里高宽可能是颠倒的
input_names = ["gemfield_in"]
output_names = ["gemfield_out"]
torch.onnx.export(DAEmodel,
                  dummy_input,
                  "syszux_scene.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

#验证使用onnx推理

import cv2
import onnxruntime
import numpy as np
import sys
import torch

from PIL import Image
from torchvision import transforms

session = onnxruntime.InferenceSession("../syszux_scene.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
print("gemfield debug required input shape", input_shape)

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC
img = cv2.resize(img, (224, 224),interpolation = cv2.INTER_LINEAR)

img = img.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
val = np.array([0.229, 0.224, 0.225])
img = (img - mean) / val
print(img)

print("gemfield debug img shape1: ",img.shape)
img= img.astype(np.float32)
img = img.transpose((2,0,1))
#img = img.transpose((2,1,0))
print("gemfield debug img shape2: ",img.shape)
img = np.expand_dims(img,axis=0)
print("gemfield debug img shape3: ",img.shape)

res = session.run([output_name], {input_name: img})
print(res)



#转换为mlmodel
from onnx_coreml import convert
model = convert(model='syszux_scene.onnx',minimum_ios_deployment_target='13')
model.save('syszux_scene.mlmodel')
