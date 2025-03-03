# MedXChat: A Unified Multimodal Large Language Model Framework towards CXRs Understanding and Generation
##### ISBI 2025 Oral

[**MedXChat: A Unified Multimodal Large Language Model Framework towards CXRs Understanding and Generation**]

[Ling Yang](https://scholar.google.com/citations?user=0x4eX9cAAAAJ&hl=zh-CN),
[Zhanyu Wang](https://scholar.google.com/citations?hl=zh-CN&user=maeFb38AAAAJ),
[Zhenghao Chen](https://scholar.google.com/citations?hl=zh-CN&user=BThVCu8AAAAJ),
Xinyu Liang,
[Luping Zhou](https://scholar.google.com/citations?user=BThVCu8AAAAJ&hl=zh-CN&oi=ao)<br/>


![teaser](assets/medxchat.png)

![teaser](assets/results.png)

## Requirements
A suitable [conda](https://conda.io/) environment named `medxchat` can be created
and activated with:

```
conda create -n medxchat python=3.9
conda activate medxchat
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

To train MedXChat, you need to download the MIMIC dataset images.

For Text-to-CXR task, the json file for MIMIC dataset is [Text-to-CXR](https://drive.google.com/file/d/12LUDdJW8_R0usXVe8EQvgCvhplje9LgH/view?usp=drive_link).

For CXR-VQA task, the json file for MIMIC dataset is [CXR-VQA](https://drive.google.com/file/d/1wh8Gi1M6AV1lH37Dnq-rAquORdg5YGCR/view?usp=drive_link).

For CXR-to-Report task, we use Chatgpt to construct instruction tuning dialogues, the json file is [CXR-to-Report](https://drive.google.com/file/d/1ZRFARG_H5odDyO0jWJNXh1x7QZ6w4QFO/view?usp=drive_link).




To train the medxchat, run
```
python train.py
```

To inference the medxchat, run
```
python inference.py
```
