import sys
sys.path.append('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/')
import torch
from diffusers import StableDiffusionPipeline
import imageio
import numpy as np
import uuid
import os


def generate_random_filename(extension=None):
    random_filename = str(uuid.uuid4())
    if extension:
        random_filename += f".{extension}"
    return random_filename


class StableDiffusion:
    def __init__(self, device):
        model_id = "/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/gvll_A100/Checkpoints/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    def generate_image_from_text(self, prompt):
        image = self.pipe(prompt=prompt).images[0]
        filename = generate_random_filename(extension="png")
        output_video_path = os.path.join("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/result_images", filename)
        image.save(output_video_path)
        return output_video_path
    
if __name__ == "__main__":
    model = StableDiffusion(device='cuda:0')
    print(model.generate_image_from_text("PA view of the chest was obtained. The lungs are clear. The hilar and cardiomediastinal contours are normal. There is no pneumothorax, pleural effusion, or consolidation."))
