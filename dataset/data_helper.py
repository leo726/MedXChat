import sys
sys.path.append('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/')
import os
import json
import numpy as np
from numpy.random import randint
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from models.mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from models.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = './Checkpoints/mplug-owl-llama-7b-video'
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.base_dir = args.base_dir
        # image_processor = MplugOwlImageProcessor.from_pretrained(args.llm_model)
        # self.tokenizer = tokenizer
        # self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
        self.choices = ['Generate a diagnostic report for this image.', 'Describe this image in detail.', 'Please provide a detailed diagnostic report of the picture.', 'Could you produce a detailed assessment of this chest xray image for me?']

    def tokenize(self, text):
        out = self.tokenizer(
            text,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=self.args.max_length)
        input_ids = out.input_ids[0]
        return input_ids


    def clean_report(self, report):
        # clean MIMIC-CXR reports
        import re
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                            .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .' 
        report = ' '.join(report.split()[:150])
        return report

    def parse(self, features):
        id = features['id']

        if 'vqa' in features:
            instruct = features['vqa']
            instruct = instruct.replace('\n', '</s>\n')
            sp = instruct.split()
            if len(sp) > 160:
                sp = sp[:160]
                instruct = " ".join(sp)
        else:
            qus = random.choice(self.choices)
            # ans = features['report'].replace('\n', '').replace('impression', "Impression").replace('findings', "Findings")
            ans = self.clean_report(features['report'])
            # ans = ans.replace("Findings", "\nFindings")
            sp = ans.split()
            # if len(sp) > 160:
            #     sp = sp[:160]
            #     ans = " ".join(sp)

            instruct = f"\nHuman:{qus}</s>\nAI:{ans}</s>"

        image_path = os.path.join(self.base_dir, features['image_path'][0])
        image = Image.open(image_path)
        # inputs = self.processor(text=[instruct], images=[Image.open(image_path)], return_tensors='pt')
       
        to_return = {
            "id": id,
            'text':instruct,
            'image':image
        }
        # to_return.update(inputs)

        return to_return


    def parse_gen(self, feature):
        feature = feature.replace('\n', '# \n')
        to_return = {
            "id": "gen_data",
            'text':feature,
        }
        return to_return 


    def transform_with_parse(self, inputs):
        if isinstance(inputs, dict):
            return self.parse(inputs)
        else:
            return self.parse_gen(inputs)
    

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.train = split == "train"
        meta = json.load(open(args.dataset, 'r'))
        if split == "train":
            self.df = meta['train']
        else:
            self.df = meta['test']
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self.parser.transform_with_parse(self.df[index])
        except Exception as e:
            print(f'Error reading for {self.df[index]["id"]}: {e}')
            idx = np.random.randint(0, len(self.df)-1)


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset



if __name__ == '__main__':
    from tqdm import tqdm
    from configs.config import parser
    args = parser.parse_args()
    loader = ParseDataset(args)

    for i in tqdm(range(loader.__len__())):
        data = loader.__getitem__(i)
        # data = loader.__getitem__(12)

        # data = loader.__getitem__(18)
        # data = loader.__getitem__(20)

