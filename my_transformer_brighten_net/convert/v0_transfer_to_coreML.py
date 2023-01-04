import sys
sys.path.append('../../my_transformer_brighten_net')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from torch import nn
import coremltools
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft 
from PIL import Image
import matplotlib.pyplot as plt

from model.autoEncoder import DenoiseAutoEncoder

DAEmodel = DenoiseAutoEncoder()
random_input = torch.rand(1, 3, 400, 600) 
modelName = "model60M.mlmodel"



def getDAEmodel():
  DAEmodel.load_state_dict(torch.load('./last_Epoch_3ae.pth',map_location="cpu"))
  DAEmodel.eval()

def prediceModel():
  print('total parameters:', sum(param.numel() for param in DAEmodel.parameters()))
  a = DAEmodel(random_input)
  print(a.shape)

def convertToCoreML():
  traced_model = torch.jit.trace(DAEmodel, random_input) 
  print("开始转换")
  model = ct.convert(
      model=traced_model,
      source="pytorch",
      inputs=[ct.ImageType(name="input_image", shape=random_input.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
      # inputs=[ct.TensorType(name="input_tensor",dtype=np.float32, shape=(1,3,400,600))],
      # outputs=[ct.TensorType(name="mid_res"), ct.ImageType(name="output_image",color_layout=ct.colorlayout.RGB)],
      outputs=[ct.TensorType(name="output_tensor",dtype=np.float32)],
      # outputs=["mid_res","output_tensor"],
      minimum_deployment_target=coremltools.target.iOS14, #Currently models that use bilinear upsampling in PyTorch can be converted to CoreML models targeting iOS 14, but not iOS 13.
      convert_to="neuralnetwork",
  ) 
  print("Set feature descriptions (these show up as comments in XCode)")
  # model.input_description["input_tensor"] = "a (1,3,96,96) shaped tensor transfered from an image"
  # model.output_description["output_tensor"] = "a (1,3,96,96) tensor, *255 and transpose((1,2,0) to convert to an image"
  model.author = "Alyosha"
  model.license = "leigithub1024/xxx"
  model.short_description = "lowlight image enhancement"
  model.version = "1.0"
  model.save(modelName)

def outputmuti255():
  print("将输出结果*255")
  spec = coremltools.utils.load_spec(modelName)
  builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
  # builder.add_elementwise(name=f"xx", input_names=[f"mid_res"], output_name=f"mid_res_xx", mode="MULTIPLY", alpha=255)
  builder.add_elementwise(name=f"multiply_xy_by_two_output_image", input_names=[f"output_tensor"], output_name=f"output_image", mode="MULTIPLY", alpha=255)
  builder.set_output(output_names=["output_image"], output_dims=[(3,400,600)])
  # builder.set_output(output_names=["output_image"], output_dims=[(3,96,96)])
  model_spec = builder.spec
  coremltools.models.utils.save_spec(model_spec, modelName)
  print(model_spec.description.output)

def outputToImage():
  # 更改模型输出为图像
  spec = coremltools.utils.load_spec(modelName)
  output = spec.description.output[0]
  output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
  output.type.imageType.height = 400
  output.type.imageType.width = 600
  output.name = 'output_image'
  coremltools.utils.save_spec(spec, modelName)
  print(spec.description.input,spec.description.output)

def prediceCoreML():
  model = coremltools.models.MLModel(modelName)
  #####输入格式：tensor
  # example_image = Image.open("1.jpg").resize((96, 96))
  # img = cv2.imread('./1.jpg')
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = cv2.resize(img, (96, 96),interpolation = cv2.INTER_LINEAR)
  # img = img.astype(np.float32) / 255.0
  # # mean = np.array([0.485, 0.456, 0.406])
  # # val = np.array([0.229, 0.224, 0.225])
  # # img = (img - mean) / val
  # print("gemfield debug img shape1: ",img.shape)
  # img= img.astype(np.float32)
  # img = img.transpose((2,0,1))
  # print("gemfield debug img shape2: ",img.shape)
  # img = np.expand_dims(img,axis=0)
  # print("gemfield debug img shape3: ",img.shape)
  # out_dict = model.predict({"input_tensor": img})

  ##### 输入格式：image
  img = Image.open('1.jpg')
  img = img.resize((600,400))
  # img = img.astype(np.float32) / 255.0
  out_dict = model.predict({"input_image": img})

  #####输出格式：tensor
  # res = out_dict["output_tensor"] 
  # arr = np.array((res[0]*255), dtype=np.uint8)
  # print("arr shape:", arr.shape)
  # arr = arr.transpose((1,2,0))
  # print("transposed_arr shape",arr.shape, arr)
  # image = Image.fromarray(arr)
  # image.show()


  ##### 输出格式：图像
  # print(out_dict)
  # output_pil_image = out_dict["output_image"]
  # image = output_pil_image.convert("RGB")
  # image_array = np.array(image)
  # image_array = image_array * 255
  # print(image_array)
  out_dict['output_image'].show()  #这里发现输出是全黑的，找到问题了，coreml的那个input格式转换没调好



#转换为onnx
def converToonnx():
  input_names = ["gemfield_in"]
  output_names = ["gemfield_out"]
  torch.onnx.export(DAEmodel,
                    random_input,
                    "syszux_scene.onnx",
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)


getDAEmodel()
prediceModel()
convertToCoreML()
outputmuti255()
outputToImage()
prediceCoreML()

# converToonnx()