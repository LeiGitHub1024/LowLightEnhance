#将pth 转为coreML
#torch version 1.12.1
print(1)
import torch
model = torch.load("autodecode.pth")
print('model loaded')
import coremltools
coreml_model = coremltools.converters.pytorch.convert(model,input_names=["input"],output_names=["output"])
print('model converted')
coreml_model.save("model.mlmodel")