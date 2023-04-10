import json
import os
import random
os.chdir('/Nlp_2023/Dialogue_Bloom/')
# from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

# model_name = "./Bloom_3BSave/"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

def joint(txt_list, m):
    joint_list = []
    for i in range(0, len(txt_list), m):
        text = '\n'.join(txt_list[i:i+m])
        joint_list.append(text)
    return joint_list


def joint_s(txt_list, m):
    joint_list = []
    for i in range(0, len(txt_list), m):
        text = '</s>'.join(txt_list[i:i+m])
        joint_list.append(text)
    return joint_list

def chunk_512(file_list):
    data_list = []
    n = 0
    for file in file_list:
        _list = json.load(open(file[0], 'r', encoding='utf-8'))[:file[1]]
        random.shuffle(_list)
        _list = _list[:file[1]]
        n += len(_list)
        data_list.extend([k for k in _list if len(k) > 300])
        list_250 = [k for k in _list if len(k) <= 300 and len(k) > 180]
        list_150 =  [k for k in _list if len(k) <= 180 and len(k) > 150]
        list_100 =  [k for k in _list if len(k) <= 150]

        data_list.extend(joint(list_250, 2))
        data_list.extend(joint(list_150, 3))
        data_list.extend(joint(list_100, 4))
    
    return data_list, n


def chunk_1024(file_list):
    data_list = []
    n = 0
    for file in file_list:
        _list = json.load(open(file[0], 'r', encoding='utf-8'))
        random.shuffle(_list)
        _list = _list[:file[1]]
        n += len(_list)
        data_list.extend([k for k in _list if len(k) > 500])
        list_400 = [k for k in _list if len(k) <= 500 and len(k) > 330]
        list_300 =  [k for k in _list if len(k) <= 330 and len(k) > 250]
        list_100 =  [k for k in _list if len(k) <= 250]

        data_list.extend(joint(list_400, 2))
        data_list.extend(joint(list_300, 3))
        data_list.extend(joint(list_100, 4))
    
    return data_list, n



def chunk_list(all_list):
    data_list = []
    for d in all_list:
        _list = d
        random.shuffle(_list)
        data_list.extend([k for k in _list if len(k) > 500])
        list_400 = [k for k in _list if len(k) <= 500 and len(k) > 330]
        list_300 =  [k for k in _list if len(k) <= 330 and len(k) > 250]
        list_100 =  [k for k in _list if len(k) <= 250]

        data_list.extend(joint_s(list_400, 2))
        data_list.extend(joint_s(list_300, 3))
        data_list.extend(joint_s(list_100, 4))
    
    return data_list

# l, n = chunk_1024(['data/bisai_dia.json', 'data/dia_data.json', 'data/kuake_data.json', 'data/ner_data.json', 'data/opqa_list1.json'])

# cut_dataset = []
# for line in l:
#     k = tokenizer.encode(line)
#     if len(k) <= 1024:
#         cut_dataset.append(line)
# print(len(l), n)
# print(len(cut_dataset))
