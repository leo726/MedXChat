from glob import glob
import json
from tqdm import tqdm
import os
from random import shuffle


# Generation
# file_path = glob('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Data/mimic/mimic_cxr/chatgpt_dialogue/*/*')

# save = []

# for i in tqdm(file_path):
#     data = json.load(open(i))
#     data = data.replace('Name', 'Dialogue').replace('Title', 'Dialogue')
#     sp = [i for i in data.split("Dialogue") if "Human" in i]
#     for i in sp:
#         dia = "\n".join(i.split('\n')[1:])
#         dia = dia.replace('Assistant', 'AI').replace('\n\n', '\n')
#         dia = dia.strip('-=\n')
#         save.append(dia)

# train = save[:7500]
# test = save[7500:]

# to_save = {'train':train, 'test':test}
# json.dump(to_save, open('chatgpt_dialogue.json', 'w'))

# --------------------VQA------------------------
# base_dir = '/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic-cxr-jpg_medvqa_v1'
# annotation = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/my_mimic_anno.json'))

# train = annotation['train']

# train_vqa = []
# for i in tqdm(train):
#     p = '/'.join(i['image_path'][0].split('/')[:-1]) + ".txt"
#     vqa_path = os.path.join(base_dir,p)
#     if os.path.isfile(vqa_path):
#         try:
#             vqa_data = json.load(open(vqa_path))
#             instruct = ''
#             for k, v in enumerate(vqa_data):
#                 instruct += f"\nHuman:{v['question']}\nAI:{v['answer']}"
#             i['vqa'] = instruct
#             train_vqa.append(i)
#         except Exception as e:
#             print(f"{vqa_path}:{e}")

# print(len(train_vqa))
# json.dump(train_vqa, open('mimic_cxr_vqa_train.json', 'w'))

# test = annotation['test']
# test_vqa = []
# for i in test:
#     p = '/'.join(i['image_path'][0].split('/')[:-1]) + ".txt"
#     vqa_path = os.path.join(base_dir,p)
#     if os.path.isfile(vqa_path):
#         try:
#             vqa_data = json.load(open(vqa_path))
#             instruct = ''
#             for k, v in enumerate(vqa_data):
#                 instruct += f"\nHuman:{v['question']}\nAI:{v['answer']}"
#             i['vqa'] = instruct
#             test_vqa.append(i)
#         except Exception as e:
#             print(f"{vqa_path}:{e}")

# print(len(test_vqa))
# json.dump(test_vqa, open('mimic_cxr_vqa_test.json', 'w'))

# val = annotation['val']
# val_vqa = []
# for i in val:
#     p = '/'.join(i['image_path'][0].split('/')[:-1]) + ".txt"
#     vqa_path = os.path.join(base_dir,p)
#     if os.path.isfile(vqa_path):
#         try:
#             vqa_data = json.load(open(vqa_path))
#             instruct = ''
#             for k, v in enumerate(vqa_data):
#                 instruct += f"\nHuman:{v['question']}\nAI:{v['answer']}"
#             i['vqa'] = instruct
#             val_vqa.append(i)
#         except Exception as e:
#             print(f"{vqa_path}:{e}")

# print(len(val_vqa))
# json.dump(val_vqa, open('mimic_cxr_vqa_val.json', 'w'))

# to_save = {'train':train_vqa, 'val':val_vqa, 'test':test_vqa}
# json.dump(to_save, open('mimic_cxr_vqa.json', 'w'))

# tiny dataset
# rg = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/my_mimic_anno.json'))
# train = rg['train']
# val = rg['val']
# test = rg['test']

# shuffle(train)
# shuffle(test)
# shuffle(val)

# train = train[:25000]
# test = test[:500]
# val = val[:500]


# vqa = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/mimic_cxr_vqa.json'))
# train_vqa = vqa['train']
# val_vqa = vqa['val']
# test_vqa = vqa['test']

# train = train + train_vqa
# test = test + test_vqa
# val = val + val_vqa

# to_save = {'train':train, 'val':val, 'test':test}



hc = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/HealthCareMagic-100k.json'))

a = hc[0]
f"Human:{a['input']} </s>\nAI:{a['output']}"

rg = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr_tiny.json'))
train = rg['train']
val = rg['val']
test = rg['test']

shuffle(train)
shuffle(test)
shuffle(val)

to_save = {'train':train, 'val':val, 'test':test}

json.dump(to_save, open('mimic_cxr_tiny.json', 'w'))



