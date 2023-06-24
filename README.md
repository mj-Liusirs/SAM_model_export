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
| absl-py              |   |  1.4.0 |
| certifi              |   |  2022.12.7 |
| charset-normalizer   |   |  2.1.1
| colorama             |   |  0.4.6
| coloredlogs          |   |  15.0.1
| contourpy            |   |  1.0.7
| cycler               |   |  0.11.0
| filelock             |   |  3.9.0
| flatbuffers          |   |  23.5.26
| fonttools            |   |  4.39.4
| humanfriendly        |   |  10.0
| idna                 |   |  3.4
| importlib-resources  |   |  5.12.0
| Jinja2               |   |  3.1.2
| kiwisolver           |   |  1.4.4
| markdown-it-py       |   |  2.2.0
| MarkupSafe           |   |  2.1.2
| matplotlib           |   |  3.7.1
| mdurl                |   |  0.1.2
| mpmath               |   |  1.2.1
| networkx             |   |  3.0
| numpy                |   |  1.23.4
| onnx                 |   |  1.10.0
| onnxruntime          |   |  1.14.1
| onnxsim              |   |  0.4.28
| opencv-python        |   |  4.7.0.72
| packaging            |   |  23.1
| Pillow               |   |  9.3.0
| pip                  |   |  23.1.2
| prettytable          |   |  3.7.0
| protobuf             |   |  3.20.3
| pycocotools          |   |  2.0.6
| Pygments             |   |  2.15.1
| pyparsing            |   |  3.0.9
| pyreadline3          |   |  3.4.1
| python-dateutil      |   |  2.8.2
| pytorch-quantization |   |  2.1.2
| PyYAML               |   |  6.0
| requests             |   |  2.28.1
| rich                 |   |  13.4.1
| scipy                |   |  1.10.1
| setuptools           |   |  41.2.0
| six                  |   |  1.16.0
| sphinx-glpi-theme    |   |  0.3
| sympy                |   |  1.11.1
| tensorrt             |   |  8.5.1.7
| torch                |   |  2.0.0+cu117
| torchaudio           |   |  2.0.2+cu117
| torchvision          |   |  0.15.2+cu117
| tqdm                 |   |  4.65.0
| typing_extensions    |   |  4.4.0
| urllib3              |   |  1.26.13
| wcwidth              |   |  0.2.6
| zipp                 |   |  3.15.0