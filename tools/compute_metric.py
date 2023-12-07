import sys
sys.path.append("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/")
from models.Xchat import XChatModel
from configs.config import parser
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
import json
import os
import torch
from tqdm import tqdm
import pickle

device = 'cuda:7'
args = parser.parse_args(args=[])
args.delta_file="/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/RG_v11/checkpoints/checkpoint_epoch5_step31594_val_loss0.965245.pth"
args.lora_inference = True
model = XChatModel(args).to(device)
tokenizer = model.tokenizer


def score(ref, hypo):
    """
    ref, dictionary of reference sentences {id:[sentence]}
    hypo, dictionary of hypothesis sentences {id:[sentence]}
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def get_prompt_embed(history):
    cnt = 0
    input_embeds = []
    for p in history:
        if p.endswith("png"):
            cnt += 1
            video = model.process_image(p)
            video = video.bfloat16()
            video = video.to(device)
            video_embs = model.prompt_wrap(video)
            input_embeds.append(video_embs)
        elif type(p) == str:
            out = model.tokenizer(p, return_tensors="pt", add_special_tokens=False)
            input_ids = out.input_ids[0]
            input_ids = input_ids.to(device)
            text_embs = model.input_embeddings(input_ids)
            text_embs = text_embs.unsqueeze(0)
            input_embeds.append(text_embs)
    embeddings = torch.cat(input_embeds, 1).to(device)
    return embeddings


def run(history):
    inputs_embeds = get_prompt_embed(history)
    with torch.no_grad():
        output = model.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=3,
            do_sample=False,
            min_new_tokens=80,
            max_new_tokens=120,
            repetition_penalty=2.0,
            length_penalty=2.0,
            temperature=0)
    sentence = tokenizer.decode(output.tolist()[0], skip_special_tokens=True)
    return sentence


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
    report = ' '.join(report.split()[:100])
    return report


def calc_score():
    hypo = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/hypo_r11_v1.json'))
    hypo = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/R2GenGPT/save/mimic_cxr/v0_deep/result/test_result.json"))
    hypo = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/hypo_r4.json"))
    hypo = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/unixgen_gen_report.json"))
    hypo = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/llm_cxr_gen.json"))
    hypo = {k:[clean_report(v[0])] for k, v in hypo.items()}
    gt = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/ref.json"))
    gt = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/unixgen_ori_report.json"))
    gt = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/llm_cxr_gt.json"))
    gt = {k:[clean_report(v[0])] for k, v in gt.items()}
    scores = score(ref=gt, hypo=hypo)
    print(scores)

data = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/my_mimic_anno.json'))
data = data['train'] + data['test'] + data['val']
data1 = {i['id']:[i['report']] for i in data}
data1 = {k:[clean_report(v[0])] for k, v in data1.items()}
gt = json.load(open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/llm_cxr_gt.json"))
gt2 = {k:data1[k] for k, v in gt.items()}
json.dump(gt2, open("/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/ft_local/llm_cxr_gt_ori.json", "w"))

# if __name__ == "__main__":
#     mimic = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/my_mimic_anno.json'))
#     test = mimic['test']
#     base_dir = "/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/images"

#     gt = {}
#     gen = {}
#     emb = {}
#     for i in tqdm(test):
#         idx = i['id']
#         gt_report = i['report'].replace('\n', '')
#         to_regress_tokens = tokenizer(
#                 gt_report,
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=100,
#                 add_special_tokens=False
#             )
#         gt_report = tokenizer.decode(to_regress_tokens['input_ids'][0], skip_special_tokens=True)
#         image_path = os.path.join(base_dir, i['image_path'][0])
#         history = [image_path, "\nHuman:Generate a diagnostic report for this image.\nAI:"]
#         embed = get_prompt_embed(history)
#         # gen_report = run(history)
#         # print(gen_report)
#         gt[idx] = [gt_report]
#         emb[idx] = embed.cpu()
#         # gen[id] = [gen_report]

#     # scores = score(ref=gt, hypo=gen)
#     pickle.dump(emb, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/inputs_embeds.pkl', 'wb'))
#     json.dump(gt, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/ref.json', 'w'))
#     # json.dump(gen, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/hypo.json', 'w'))
#     # json.dump(scores, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/scores.json', 'w'))
#     # print(scores)


def decode(output_token):
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0].strip()
    output_text = output_text.split('\n')[0].strip()
    output_text = output_text.replace('<unk>', '')
    return output_text


# calc_score()


if __name__ == '__main__':

    def get_n_from_item(item):
        key, tensor = item
        return tensor.shape[1]
    
    inputs_embeds = pickle.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/inputs_embeds.pkl', 'rb'))
    ref = json.load(open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/ref.json', 'r'))
    ref = {k:[clean_report(v[0])] for k,v in ref.items()}
    sorted_items = sorted(inputs_embeds.items(), key=get_n_from_item)
    names = [key for key, value in sorted_items]
    embes = [value for key, value in sorted_items]
    del inputs_embeds

    all_names = []
    all_predict = []

    # names = list(inputs_embeds.keys())
    # embes = list(inputs_embeds.values())
    nms = []
    ems = []
    for i, name in tqdm(enumerate(names)):
        # if i> 500:
        #     break
        if i == 0:
            nms.append(name)
            ems.append(embes[i])
            size = embes[i].shape[1]
        elif embes[i].shape[1] == size:
            nms.append(name)
            ems.append(embes[i])
            if len(ems) == 16:
                with torch.no_grad():
                    output = model.model.language_model.generate(
                        inputs_embeds=torch.cat(ems, 0).to(device),
                        num_beams=3,
                        do_sample=False,
                        min_new_tokens=80,
                        max_new_tokens=120,
                        repetition_penalty=2.0,
                        length_penalty=2.0,
                        temperature=0)
                hypo = [decode(i) for i in output]
                all_names.extend(nms)
                all_predict.extend(hypo)
                nms = []
                ems = []   
        else:
            size = embes[i].shape[1]
            if len(ems):
                with torch.no_grad():
                        output = model.model.language_model.generate(
                            inputs_embeds=torch.cat(ems, 0).to(device),
                            num_beams=3,
                            do_sample=False,
                            min_new_tokens=80,
                            max_new_tokens=120,
                            repetition_penalty=2.0,
                            length_penalty=2.0,
                            temperature=0)
                hypo = [decode(i) for i in output]
                print(hypo)
                all_names.extend(nms)
                all_predict.extend(hypo)
                nms = []
                ems = []
            nms.append(name)
            ems.append(embes[i])

    if len(ems):
        with torch.no_grad():
                output = model.model.language_model.generate(
                    inputs_embeds=torch.cat(ems, 0).to(device),
                    num_beams=3,
                    do_sample=False,
                    min_new_tokens=80,
                    max_new_tokens=120,
                    repetition_penalty=2.0,
                    length_penalty=2.0,
                    temperature=0.3)
        hypo = [decode(i) for i in output]
        all_names.extend(nms)
        all_predict.extend(hypo)

    print(len(all_names))
    hypo = {k:[v] for k,v in zip(all_names, all_predict)}
    ref = {k:v for k,v in ref.items() if k in hypo}
    json.dump(hypo, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/hypo_r11_v1.json', 'w'))
    scores = score(ref=ref, hypo=hypo)
    # json.dump(hypo, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/hypo.json', 'w'))
    json.dump(scores, open('/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/tools/testing_result/scores_r11_v1.json', 'w'))

