import torch
import os
os.chdir('/workspace/Nlp_2023/Dialogue_Bloom/')

# 加载模型的权重
state_dict = torch.load('/workspace/Model_WinGPT_pretrain/GPT-base-13B/epoch2/pytorch_model.bin')

# 将权重转为16位
state_dict_half = {k: v.half() for k, v in state_dict.items()}

# 保存权重
torch.save(state_dict_half, '/workspace/Model_WinGPT_pretrain/GPT-base-13B/epoch2/pytorch_half.bin')
