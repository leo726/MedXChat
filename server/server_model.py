import sys
sys.path.append("/mnt/sdc/yangling/MedXchat/")
import torch
import torch.nn.functional as F
from typing import List, Tuple
from models.Xchat import XChatModel
from dataset.data_helper import FieldParser
from configs.config import parser
from models.sd import StableDiffusion
from models.cxr_diffusion.test import CXRDiffusion
import gradio as gr


device = 'cuda:0'
device_video = 'cuda:0'

# V2: <video>generate a video</video>
# V3: <|video|>

args = parser.parse_args(args=[])
# port 80
# args.delta_file="/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/VQA_v11/checkpoints/checkpoint_epoch8_step14840_val_loss0.764227.pth"
args.delta_file="/mnt/sdc/yangling/MedXchat/server/checkpoint_epoch1_step714856_val_loss0.863923.pth"
# args.delta_file="/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/T2V_v11/checkpoints/checkpoint_epoch3_step936_val_loss0.767294.pth"
args.lora_inference = True

model = XChatModel(args).to(device)
sd_model = CXRDiffusion(device=device_video)
parser = FieldParser(args)
tokenizer = model.tokenizer

def add_video_idx(cnt, video_emb):
    out = model.tokenizer(f"this is video {str(cnt)}", return_tensors="pt", add_special_tokens=False)
    input_ids = out.input_ids[0]
    input_ids = input_ids.to(device)
    text_embs = model.input_embeddings(input_ids)
    text_embs = text_embs.unsqueeze(0)
    video_emb = torch.cat([text_embs, video_emb], dim=1)
    return video_emb


# prefix = model.tokenizer("The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions and should refuse to respond to any text or video that incites pornography, hatred, violence, profanity or abuse against an individual or group.", return_tensors="pt", add_special_tokens=False)
# prefix_ids = prefix.input_ids[0]
# prefix_ids = prefix_ids.to(device)
# prefix_embs = model.input_embeddings(prefix_ids)
# prefix_embs = prefix_embs.unsqueeze(0)

def get_prompt_embed(history):
    cnt = 0
    input_embeds = []
    # input_embeds.append(prefix_embs)   

    for prompt in history:
        for pro in prompt:
            for p in pro:
                if p is None:
                    continue
                if p.endswith("png"):
                    cnt += 1
                    video = model.process_image(p)
                    video = video.bfloat16()
                    video = video.to(device)
                    video_embs = model.visual_encoder(video)
                    video_embs = video_embs.to(device)
                    # video_embs = add_video_idx(cnt, video_embs)
                    input_embeds.append(video_embs)
                elif type(p) == str:
                    out = model.tokenizer(p, return_tensors="pt", add_special_tokens=False)
                    input_ids = out.input_ids[0]
                    input_ids = input_ids.to(device)
                    text_embs = model.input_embeddings(input_ids)
                    text_embs = text_embs.unsqueeze(0)
                    input_embeds.append(text_embs)
    
    out = model.tokenizer('\nAI:', return_tensors="pt", add_special_tokens=False)
    input_ids = out.input_ids[0]
    input_ids = input_ids.to(device)
    text_embs = model.input_embeddings(input_ids)
    text_embs = text_embs.unsqueeze(0)
    input_embeds.append(text_embs)
    input_embeds = input_embeds[-3:]
    embeddings = torch.cat(input_embeds, 1).to(device)
    return embeddings


def gen_video(prompt):
    gen_video_path = sd_model.generate_image_from_text(prompt=prompt)
    return gen_video_path


def run(video_decoder, history, max_len=50, temperature=0.0, top_p=1.0):
    embeddings = get_prompt_embed(history)
    if embeddings.shape[1] > max_len:
        embeddings = embeddings[:, -max_len:]

    first_flag = True
    gen_video_flag = False
    hold_flag = True
    with torch.no_grad():
        out = None
        out_2 = None
        prompt = None
        for i in range(240):
            output = model.model.language_model(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
            logits = output.logits[:, -1, :]  # (N, vocab_size)
            next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
            next_token = next_token.long().to(embeddings.device)

            # First part
            if out is None:
                out = next_token
            elif out is not None and model.tokenizer.decode(next_token[0]) != '<' and first_flag:
                out = torch.cat([out, next_token], dim=-1)
            else:
                first_flag = False
            
            # Second part
            if not first_flag and not gen_video_flag:
                if model.tokenizer.decode(next_token[0]) == "</":
                    prompt_text = model.tokenizer.decode(prompt[0])
                    prompt_text = prompt_text.replace("<video>", "")
                    video_path = gen_video(prompt_text)
                    gen_video_flag = True

                if prompt is None:
                    prompt = next_token
                else:
                    prompt = torch.cat([prompt, next_token], dim=-1)
                    prompt_text = model.tokenizer.decode(prompt[0])
                    print(prompt_text)
            
            if not first_flag and gen_video_flag:
                if model.tokenizer.decode(next_token[0]) not in ["</", "video", ">"] and hold_flag:
                    hold_flag = False
                    if out_2 is None:
                        out_2 = next_token
                    else:
                        out_2 = torch.cat([out_2, next_token], dim=-1)


            next_embedding = model.input_embeddings(next_token)
            embeddings = torch.cat([embeddings, next_embedding], dim=1)

            if model.tokenizer.decode(next_token[0]) in ["</s>", "Hu",  "Human", "#", "Human:", ".</s>"]:
                print(f"break at {i} with {model.tokenizer.decode(next_token[0])}")
                break
            
            if not gen_video_flag:
                out_text = model.tokenizer.decode(out[0])
                print(out_text)
                yield (out_text, None, None)
            else:
                if out_2 is not None:
                    out_text_2 = model.tokenizer.decode(out_2[0])
                    print(out_text_2)
                    yield (out_text, video_path, out_text_2)
                else:
                    yield (out_text, video_path, None)

        # outputs = model.model.language_model.generate(
        #     inputs_embeds=embeddings,
        #     num_beams=3,
        #     do_sample=True,
        #     max_new_tokens=100,
        #     # repetition_penalty=self.hparams.repetition_penalty,
        #     # length_penalty=self.hparams.length_penalty,
        #     temperature=0.0,
        # )
        return


if __name__ == '__main__':
    state = [(['check this out', '/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/gvll_A100/results_videos/f84f49f1-6482-45aa-a8c3-0f21adf9c896.mp4'], [None])]
    response = run(state)
    for chunk in response:
        text = chunk[0]
        out_video = chunk[1]
        print(text)
        print(out_video)