# SAM_model_export
Export SAM encoder(image embedding) and decoder(prompt) into onnx and trt model

## 将SAM模型导出成TensorRT格式推理（通过onnx中转）
运行python脚本export_onnx_trt.py
main.cpp中是c++运行trt示例

## 将SAM模型导出成Onnx格式推理
运行python脚本export_onnx_only.py
与export_onnx_trt中使用的模型输入尺寸不同

- 下载SAM模型 `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)放在downloaded文件夹内
- 安装torch==2.0.0+cu117：[下载torch-2.0.0+cu117-cp38-cp38-win_amd64.whl](https://download.pytorch.org/whl/torch_stable.html)（cp38表示python版本3.8，amd64表示64位）
- 执行 **pip install torch-2.0.0+cu117-cp38-cp38-win_amd64.whl**

python依赖库在以下版本测试通过

| Package                     |     | Version                                                                             |
| -------------------------------|---- | ----------------------------------------------------------------------------------- |
| numpy                |   |  1.23.4
| onnx                 |   |  1.10.0
| onnxruntime          |   |  1.14.1
| onnxsim              |   |  0.4.28
| opencv-python        |   |  4.7.0.72
| scipy                |   |  1.10.1
| tensorrt             |   |  8.5.1.7
| torch                |   |  2.0.0+cu117
| torchaudio           |   |  2.0.2+cu117
| torchvision          |   |  0.15.2+cu117