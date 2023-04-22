import json
import random
import os
os.chdir('./Nlp_2023/Medical_data/')
'''
书籍：51本，
药品说明书：9700
疾病说明书：7200
百科: 50000


实体识别：18000
事件抽取：15000

报告诊断：20000
问诊对话：60000  以\t隔开
报告生成：8000 以\t隔开

单轮问答：kuake 36000 + llama 7000 + 书籍 2000  + bingli 10000    [100000]
多轮医疗问答： 50000  以\t隔开
多项选择:50000 

'''
from transformers import AutoTokenizer,BloomTokenizerFast

model_name = "./Model_TH/Bloom_6B4_zh/"
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

def joint(txt_list, join_length):
    new_txtlist = []
    point = []
    n = 0
    while len(point) < len(txt_list) and n < len(txt_list):
        if n not in point:
            joint_str = str(txt_list[n])
            point.append(n)

            for j in range(n+1, len(txt_list)):
                if j not in point:
                    if len(joint_str + '\t' + str(txt_list[j])) > join_length:
                        pass
                    else:
                        joint_str +=  '\t' + str(txt_list[j])
                        point.append(j)
            new_txtlist.append(joint_str)
        else:
            pass

        n += 1
    return new_txtlist


def chunk_1024(file_list, length):
    data_list = []
    n = 0
    _list = []
    for file in file_list:
        _list += file[0][:file[1]]
    random.shuffle(_list)
    n += len(_list)
    print(n)
    data_list.extend([k for k in _list if len(k) >= length])
    '''
    时间太长了，将列表分100批进行组合
    '''
    list_1080 = [k for k in _list if len(k) < length]
    length = len(list_1080)
    slice = 100
    m = int(length/slice)+1
    for i in range(slice):
        slice_list = list_1080[i*m:(i+1)*m]
        joint_1080 = joint(slice_list, length)
        data_list.extend(joint_1080)
    
    return data_list, n


def chunk_2048(file_list, length):
    data_list = []
    n = 0
    _list = []
    for file in file_list:
        _list += file[0][:file[1]]
    random.shuffle(_list)
    n += len(_list)
    print(n)
    data_list.extend([k for k in _list if len(k) >= length])
    '''
    时间太长了，将列表分100批进行组合
    '''
    list_2100 = [k for k in _list if len(k) < length]
    length = len(list_2100)
    slice = 300
    m = int(length/slice)+1
    for i in range(slice):
        slice_list = list_2100[i*m:(i+1)*m]
        joint_2100 = joint(slice_list, length)
        data_list.extend(joint_2100)
    
    return data_list, n



def sliding_window_template_with_examples(text, length, step):
    left = 0
    right = length
    _list = []
    while right < len(text) +step:
        line = text[left:right]
        _list.append(line)
        left += step
        right += step
    return _list

'''
doc
'''
doc_list = json.load(open('./new_data/doc_data/bookdoc.json', 'r', encoding='utf-8')) + json.load(open('./new_data/doc_data/drug_disease_json.json', 'r', encoding='utf-8')) 

erton =  [str(k).strip().replace('\n', '')[1:-1] for k in  json.load(open('./new_data/doc_data/erton_zd.json', 'r', encoding='utf-8'))]
siyuan = [str(k).strip().replace('\n', '')[1:-1] for k in  json.load(open('./new_data/doc_data/siyuan_zd.json', 'r', encoding='utf-8'))]
zs_zd = [str(k).strip().replace('\n', '')[1:-1] for k in  json.load(open('./new_data/doc_data/zs_zd.json', 'r', encoding='utf-8'))]
bc_doc = json.load(open('./new_data/doc_data/bc_info.json', 'r', encoding='utf-8'))
ruyuan = [str(k).strip().replace('\n', '') for k in  json.load(open('./new_data/doc_data/RYjilu_zd.json', 'r', encoding='utf-8'))]
chuyuan = [str(k).strip().replace('\n', '') for k in  json.load(open('./new_data/doc_data/CYjilu_zd.json', 'r', encoding='utf-8'))]
ssjl = json.load(open('./new_data/doc_data/ssjl_data.json', 'r', encoding='utf-8'))

# doc_medical,_ = chunk_2048([[bc_doc,100000],
#                            [erton, 100000],
#                            [siyuan, 100000],
#                            [zs_zd, 100000],
#                            [ruyuan, 100000],
#                            [chuyuan,100000],
#                            [ssjl, 100000]], 2250)                         

# doc_2048 = []
# for k in (doc_list + doc_medical):
#     if len(k) <= 2250:
#         doc_2048.append(k)
#     else:
#         line_list = sliding_window_template_with_examples(k, 2250, 2100)
#         doc_2048.extend(line_list)

# print(doc_2048[-1])
# print(len(doc_2048))
# print(len(''.join(doc_2048)))

# pre_data = ['<s>'+k+'</s>' for k in  doc_2048]
# cut_dataset = []
# max_tok = []
# for line in pre_data:
#     k = tokenizer.encode(line)
#     if len(k) <= 2048:
#         max_tok.append(len(k))
#         cut_dataset.append(line)

# print(len(cut_dataset))
# print(max(max_tok))
# print(len(''.join(cut_dataset)))

# en_train_data = open('./pretrain_data/doc_2048.json', 'wb')
# eachline = json.dumps(cut_dataset, ensure_ascii=False, indent=2) + '\n'
# eachline = eachline.encode()
# en_train_data.write(eachline)

# 206930945  230714
# 194465211  111016



'''
dia
'''
dia_hdf = [''.join(k) for k in  json.load(open('./new_data/dia_data/dia_hdf.json', 'r', encoding='utf-8'))]
dia_report = [''.join([str(j) for j in k]) for k in  json.load(open('./new_data/dia_data/dia_report.json', 'r', encoding='utf-8'))]

dia_tc = [''.join(k) for k in  json.load(open('./new_data/dia_data/dia_tc.json', 'r', encoding='utf-8'))]

dia_imcs = [''.join([str(j) for j in k]) for k in  json.load(open('./new_data/dia_data/imcs_report.json', 'r', encoding='utf-8'))]
print(dia_imcs[-1])
dia_medical = []
med_dia =  [''.join(k) for k in  json.load(open('./new_data/dia_data/medical_dia.json', 'r', encoding='utf-8'))]
for x in med_dia:
    x = x.replace('User:', '患者:')
    x = x.replace('System:', '医生:')
    dia_medical.append(x)

print(dia_medical[1000])
belle_med =  json.load(open('./new_data/dia_data/belle_med_dia.json', 'r', encoding='utf-8'))
belle_other = json.load(open('./new_data/dia_data/belle_other_dia.json', 'r', encoding='utf-8'))
random.shuffle(belle_other)
belle_multi = []
for key in (belle_med + belle_other[:60000]):
    kkk = ''.join((key['instruction']+key['output']))
    kkk = kkk.replace('\nHuman:', '||Human:')
    kkk = kkk.replace('\nAssistant:', '||AIer:')
    belle_multi.append(''.join(kkk.split('||')))


print(belle_multi[-1])
'''
ie
'''
ner = ['实体识别:'+k['text']+str(k['answer']) for k in  json.load(open('./new_data/ie_data/ner_data.json', 'r', encoding='utf-8'))]
event = ['事件提取:'+k['text']+str(k['event']) for k in  json.load(open('./old_data/ie_data/CHIP-CDEE_train.json', 'r', encoding='utf-8'))]

'''
multi_choice
'''
op_all = []
opqa = json.load(open('./new_data/multi_qa/multi_med_qa.json', 'r', encoding='utf-8'))
for op in opqa:
    ques = op['question']
    options = op['options']
    answer = op['answer']
    h = '\n'.join([k+ '.'+ options[k] for k in options])
    op_all.append(ques+'\n'+h+'\n'+ '答案:' +answer)

med_choice = json.load(open('./new_data/multi_qa/medical_choice.json', 'r', encoding='utf-8'))
for mc in med_choice:
    cat = mc['category']
    ques = mc['question']
    options = mc['options']
    answer = mc['answer']
    mh = '\n'.join([k for k in options])
    ana = mc['analysis']
    if len(ana) > 0:
        op_all.append(cat+'\n'+ques+'\n'+mh+'\n'+ '答案:' +answer + '\n'+ana)
    else:
        op_all.append(cat+'\n'+ques+'\n'+mh+'\n'+ '答案:' +answer)

print(op_all[-1])
'''
qa
'''

cot_med_qa = [k['instruction']+k['input']+k['output'] for k in  json.load(open('./new_data/qa_data/cot_med_qa.json', 'r', encoding='utf-8'))]

cot_other_qa = [k['instruction']+k['input']+ k['output'] for k in  json.load(open('./new_data/qa_data/cot_other_qa.json', 'r', encoding='utf-8'))]
random.shuffle(cot_other_qa)
print(cot_med_qa[-1])
kuake_qa = [k['text']+ k['answer'] for k in  json.load(open('./new_data/qa_data/kuake_qa.json', 'r', encoding='utf-8'))]
llama_qa = [k['text']+k['answer'] for k in  json.load(open('./new_data/qa_data/llama_qa.json', 'r', encoding='utf-8'))]
print(kuake_qa[-1])
bingli_qa = [k['instruction']+k['input']+k['output'] for k in  json.load(open('./new_data/doc_data/medical_bingli.json', 'r', encoding='utf-8'))]

lora_qa = [k['instruction']+k['input']+k['output'] for k in  json.load(open('./new_data/qa_data/trans_chinese_alpaca_data.json', 'r', encoding='utf-8'))]
print(lora_qa[-2])

# dia_chunk, _ = chunk_2048([[dia_hdf, 100000],
#                            [dia_report, 100000],
#                            [dia_tc, 100000],
#                            [dia_imcs, 100000],
#                            [dia_medical, 100000],
#                            [belle_multi, 100000],
#                            [ner,100000],
#                            [event, 100000],
#                            [op_all,100000],
#                            [cot_med_qa,100000],
#                            [cot_other_qa,60000],
#                            [kuake_qa,100000],
#                            [llama_qa,100000],
#                            [bingli_qa,100000],
#                            [lora_qa, 100000]], 2800)


# chunking_2048 = []
# for k in dia_chunk:
#     if len(k) <= 2800:
#         chunking_1024.append(k)
#     else:
#         line_list = sliding_window_template_with_examples(k, 2800, 2650)
#         chunking_2048.extend(line_list)

# pre_data = ['<s>'+k+'</s>' for k in  chunking_2048]
# print(len(pre_data))

# cut_dataset = []
# max_tok = []
# for line in pre_data:
#     k = tokenizer.encode(line)
#     if len(k) <= 2048:
#         max_tok.append(len(k))
#         cut_dataset.append(line)

# print(len(cut_dataset))
# print(max(max_tok))
# print(len(''.join(cut_dataset)))

# en_train_data = open('./pretrain_data/diaqa_data_2048.json', 'wb')
# eachline = json.dumps(cut_dataset, ensure_ascii=False, indent=2) + '\n'
# eachline = eachline.encode()
# en_train_data.write(eachline)

#159257740  124178
#148820969  53292


'''
kg
'''
# kg_data =  ''.join([k.strip('\n')  for k in open('./new_data/kg/surgery.txt', 'r', encoding='utf-8')] )+ ''.join([k.strip('\n') for k in open('./new_data/kg/subject_disease.txt', 'r', encoding='utf-8')])
# kg_data =kg_data.split('全文下载')

# kg_cmekg = [k['text']+k['answer'] for k in json.load(open('./new_data/kg/cmekg_dis_train_data.json', 'r', encoding='utf-8'))] + [k['text']+k['answer'] for k in json.load(open('./new_data/kg/cmekg_lab_train_data.json', 'r', encoding='utf-8'))]  + [k['text']+k['answer'] for k in json.load(open('./new_data/kg/cmekg_sym_train_data.json', 'r', encoding='utf-8'))]

# kg_chunk,_ = chunk_2048([[kg_data,100000],
#                          [kg_cmekg, 100000]], 2600)

# kg_list = []
# for line in kg_chunk:
#     if len(line) < 2600:
#         kg_list.append(line)
#     else:
#         kg_list.extend(sliding_window_template_with_examples(line, 2600, 2450))

# print(len(kg_list))
# kg_dataset = []
# long_chunk = ['<s>'+k+'</s>' for k in kg_list]
# max_tok = []
# for line in long_chunk:
#     k = tokenizer.encode(line)
#     if len(k) <= 2048:
#         max_tok.append(len(k))
#         kg_dataset.append(line)

# print(len(kg_dataset))
# print(max(max_tok))
# print(len(''.join(kg_dataset)))
# #8307660
# en_train_data = open('./pretrain_data/pretrain_kg_2048.json', 'wb')
# eachline = json.dumps(kg_dataset, ensure_ascii=False, indent=2) + '\n'
# eachline = eachline.encode()
# en_train_data.write(eachline)
#6408
#3033
