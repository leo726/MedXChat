import os
import sys
sys.path.append('/mnt/sdc/yangling/MedXchat/')
from models.Xchat_qwenvl import XChatModel
from configs.config_qwen import parser
import torch
from PIL import Image
import json

def clean_report(report):
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
    
if __name__ == '__main__':
    ckpt_file = ''  #checkpoints.pth
    state_dict = torch.load(ckpt_file, map_location='cpu')['model']
    args = parser.parse_args()
    medxchat = XChatModel(args).to('cuda:0')
    medxchat.load_state_dict(state_dict=state_dict, strict=False)
    
    report = {}
    generate = {}
    count = 0
    
    path = 'data/annotation.json'
    data = json.load(open(path))
    for i in data['test']:
        # print(i)
        id = i['id']
        path1 = 'mimic_cxr/images/' + i['image_path'][0]
        query = medxchat.tokenizer.from_list_format([
        {'image': path1},
        {'text': 'Generate a diagnostic report for this image.'},
    ])
    #     query = medxchat.tokenizer.from_list_format([
    #     {'image': path1},
    #     {'text': 'Outline the  of the disease in the image.'},
    # ])
        response, history = medxchat.model.chat(medxchat.tokenizer, query, history=None)
        # print(response)
        report[id] = [clean_report(i['report'])]
        generate[id] = [response]
        count += 1
        print(count)
   
        
    report_str = json.dumps(report,indent=4)
    with open('report.json', 'w') as report_file:
        report_file.write(report_str)
    
    generate_str = json.dumps(generate,indent=4)
    with open('generate.json', 'w') as generate_file:
        generate_file.write(generate_str)
