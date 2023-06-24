import torch
import numpy as np
import os
import cv2
import warnings
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# Generate preprocessing model of Segment-anything in onnx format
# Target image size is 1024x720
image_size = (1024, 1024)
sam_checkpoint = "downloaded/sam_vit_b_01ec64.pth"
model_type = 'vit_b'
output_path = 'sam_preprocess.onnx'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda')
transform = ResizeLongestSide(sam.image_encoder.img_size)

# image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
image = cv2.imread('downloaded/example_img.jpg')
input_image = transform.apply_image(image)
print("input_image shape:",input_image.shape)
input_image_torch = torch.as_tensor(input_image, device='cuda')
input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]


class Model(torch.nn.Module):
    def __init__(self, image_size, checkpoint, model_type):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)
        self.image_size = image_size

    def forward(self, x):
        self.predictor.set_torch_image(x, (self.image_size))
        return self.predictor.get_image_embedding()

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

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    model = Model(image_size, sam_checkpoint, model_type)
    model_trace = torch.jit.trace(model, input_image_torch)
    # torch.onnx.export(
    #     model_trace, 
    #     input_image_torch, 
    #     output_path,
    #     export_params=True,
    #     verbose=False,
    #     opset_version=13,
    #     do_constant_folding=True,
    #     input_names=['input'], 
    #     output_names=['output']
    # )

    export_sam_model()
