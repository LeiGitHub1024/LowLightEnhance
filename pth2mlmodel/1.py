# from onnx_coreml import convert
import onnx_coreml
import coremltools as ct
# ct.converters.onnx.convert()

print(1)

model = ct.converters.onnx.convert(
  model='syszux_scene.onnx',
  minimum_ios_deployment_target='13'

  )
model.save('syszux_scene.mlmodel')