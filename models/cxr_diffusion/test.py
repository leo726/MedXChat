import sys
sys.path.append("/models/cxr_diffusion")
from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyInference
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import numpy as np
import torch
import torchvision
from PIL import Image
import uuid
import os


def generate_random_filename(extension=None):
    random_filename = str(uuid.uuid4())
    if extension:
        random_filename += f".{extension}"
    return random_filename


class CXRDiffusion:
    def __init__(self, device):
        self.resume_path = 'finetune_prompt2cxr.ckpt'
        self.batch_size = 1
        self.logger_freq = 1
        self.learning_rate = 1e-5
        self.model = create_model('/models/cxr_diffusion/models/cldm_v21.yaml').cpu()
        self.model.load_state_dict(load_state_dict(self.resume_path, location='cpu'))
        self.model.learning_rate = self.learning_rate
        self.model.sd_locked = False
        self.model.only_mid_control = False
        self.model.to(torch.device(device))


    def generate_image_from_text(self, prompt):
        # create dataset
        x = torch.rand(1,256,256,3)
        batch = {'jpg':x,'txt':[prompt],'hint':x}
        log = self.model.log_images(batch)
        grid = torchvision.utils.make_grid(log['samples_cfg_scale_1.10'], nrow=4)
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = generate_random_filename(extension="png")
        output_path = os.path.join("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/result_images", filename)
        Image.fromarray(grid).save(output_path)
        return output_path

# model = cxr_diffusion(device='cuda:0')

# img_path = model.generate_image_from_text('Lateral view of the chest was obtained. The previously seen multifocal bibasilar airspace opacities have almost completely resolved with only slight scarring seen at the bases. There are new ill-defined bilateral linear opacities seen in the upper lobes, which given their slight retractile behavior are likely related to radiation fibrosis.')
# print(img_path)
