import torch
import numpy as np
from typing import Tuple
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms.functional import resize, to_pil_image
from torch.nn import functional as F
import cv2
import warnings


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


# @torch.no_grad()
def pre_processing(image: np.ndarray, target_length: int, device,pixel_mean,pixel_std,img_size):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = np.array(resize(to_pil_image(image), target_size))

    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch

def export_embedding_model(onnx_model_path = 'vit_b_embedding_gpu.onnx'):
    sam_checkpoint = "downloaded/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    image = cv2.imread('downloaded/example_img.jpg')
    target_length = sam.image_encoder.img_size
    pixel_mean = sam.pixel_mean 
    pixel_std = sam.pixel_std
    img_size = sam.image_encoder.img_size
    input_image_torch = pre_processing(image, target_length, device,pixel_mean,pixel_std,img_size)


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            sam.image_encoder,
            input_image_torch,
            onnx_model_path,
            export_params=True,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["image_embeddings"]
        )    

def export_sam_model(onnx_model_path = "sam.onnx"):
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = "downloaded/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float)
        # "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    # output_names = ["masks", "iou_predictions", "low_res_masks"]
    output_names = ["masks", "scores"]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )  


def export_engine_image_encoder(f='vit_b_embedding_gpu.onnx', half=True):
    import tensorrt as trt
    from pathlib import Path
    file = Path(f)
    f = file.with_suffix('.engine')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 6
    print("workspace: ", workspace)
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    half = True
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())


def export_engine_prompt_encoder_and_mask_decoder(f='sam.onnx', half=True):
    import tensorrt as trt
    from pathlib import Path
    file = Path(f)
    f = file.with_suffix('.engine')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 6
    print("workspace: ", workspace)
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    profile = builder.create_optimization_profile()
    profile.set_shape('image_embeddings', (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
    profile.set_shape('point_coords', (1, 2,2), (1, 5,2), (1,10,2))
    profile.set_shape('point_labels', (1, 2), (1, 5), (1,10))
    profile.set_shape('mask_input', (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
    profile.set_shape('has_mask_input', (1,), (1, ), (1, ))
    # profile.set_shape_input('orig_im_size', (416,416), (1024,1024), (1500, 2250))
    # profile.set_shape_input('orig_im_size', (2,), (2,), (2, ))
    config.add_optimization_profile(profile)

    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())


if __name__ == '__main__':
    with torch.no_grad():
        export_embedding_model('vit_b_embedding_gpu.onnx')
        export_sam_model("sam.onnx")

        #加载过import tensorrt as trt后，上面的torch.exportOnnx会失败，需要按此顺序执行
        export_engine_image_encoder('vit_b_embedding_gpu.onnx')  
        export_engine_prompt_encoder_and_mask_decoder("sam.onnx")



        

        

