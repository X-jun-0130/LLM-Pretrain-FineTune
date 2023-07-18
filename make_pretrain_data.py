import json
import random
import os
from transformers import LlamaTokenizer


tokenizer_llama = LlamaTokenizer.from_pretrained('/Model_TH/openllama-7b/')


def joint(txt_list):
    
    new_txtlist = []
    point = []
    n = 0
    while len(point) < len(txt_list) and n < len(txt_list):
        if n not in point:
            joint_str = txt_list[n][0]
            length_joint = txt_list[n][1]
            point.append(n)
            for j in range(n+1, len(txt_list)):
                if j not in point:
                    if (length_joint + txt_list[j][1]) >= 2000 and (length_joint + txt_list[j][1]) <= 2048:
                        point.append(j)
                        joint_str +=  txt_list[j][0]
                        break
                    elif (length_joint + txt_list[j][1]) < 2000:
                        joint_str +=  txt_list[j][0]
                        length_joint += txt_list[j][1]
                        point.append(j)
                    else:
                        pass
            new_txtlist.append(joint_str)
        else:
            pass

        n += 1
    return new_txtlist

# def joint(txt_list):
#     new_txtlist = []
#     while len(txt_list) >= 2:
#         point = []
#         joint_str = ''
#         joint_str = txt_list[0][0]
#         length_joint = txt_list[0][1]
#         point.append(0)
#         for j in range(1, len(txt_list)):
#             if (length_joint + txt_list[j][1]) >= 2000 and (length_joint + txt_list[j][1]) <= 2048:
#                 point.append(j)
#                 joint_str +=  txt_list[j][0]
#                 break
#             elif (length_joint + txt_list[j][1]) < 2000:
#                 joint_str +=  txt_list[j][0]
#                 length_joint += txt_list[j][1]
#                 point.append(j)
#             else:
#                 pass

#         new_txtlist.append(joint_str)
#         new_txt_list = [txt_list[z] for z in range(len(txt_list)) if z not in point]
        
#         if len(new_txt_list) == 1:
#             new_txtlist.append(new_txt_list[0][0])
#         elif len(new_txt_list) > 1:
#             txt_list = new_txt_list
#             continue
#         else:
#             break
#     return new_txtlist


from concurrent.futures import ProcessPoolExecutor
def mp_process(function,txt_list):
    select_all = []
    with ProcessPoolExecutor(max_workers=32) as Pool:
        results = Pool.map(function, txt_list)
        for result in results:
            select_all.extend(result)
    return select_all


def get_chunk(_list, length):
    data_list, list_chunk = [], []
    for k_line in _list:
        if k_line[1] > length:
            data_list.append(k_line)
        else:
            list_chunk.append(k_line)

    print('切分错误:', str(len(data_list)))
    cut_list = []
    for ml in data_list:
        cut_list.extend(sliding_window_template_with_examples(ml, int(0.5*len(ml)), int(0.5*len(ml))-75))

    '''
    时间太长了，将列表分1000批进行组合
    '''
    list_chunk += cut_list
    random.shuffle(list_chunk)
    list_length = len(list_chunk)
    slice = 10
    m = int(list_length/slice)+1
    list_all = []
    result_all = []
    for i in range(slice):
        print('拼接:',str(i))
        slice_list = list_chunk[i*m:(i+1)*m]
        # list_all.append(slice_list)
        joint_chunk = joint(slice_list)
        result_all.extend(joint_chunk)
    
    # result_all = mp_process(joint, list_all)
    
    return result_all


def sliding_window_template_with_examples(text, length, step):
    left = 0
    right = length
    _list = []
    while right < len(text) + step:
        line = text[left:right]
        _list.append(line)
        left += step
        right += step
    return _list


def cut_sentence(datalist):
    data_cut = []
    for k in datalist:
        token_leng = len(tokenizer_llama.encode(k))
        if token_leng <= 2047:
            data_cut.append([k+'</s>', token_leng+1])
        else:
            x = int(round(token_leng/2047)) + 1  #向上取整
            length = int(len(k) / x)
            line_list = sliding_window_template_with_examples(k, length, length-75)
            for line in line_list:
                len_line = len(tokenizer_llama.encode(line))
                if len_line <= 2047:
                    data_cut.append([line+'</s>', len_line+1])
                else:
                    llen = int(len(line)/2)
                    _line_list = sliding_window_template_with_examples(line, llen, llen-75)
                    for ll in _line_list:
                        data_cut.append([ll+'</s>', len(tokenizer_llama.encode(ll))+1]) 
    return data_cut

def get_pretrain_text(data_list):
    x = 32
    y = int(len(data_list)/x)+1
    x_all = []
    for i in range(x):
        slice_list = data_list[i*y:(i+1)*y]
        x_all.append(slice_list)

    cut_list = mp_process(cut_sentence, x_all)
    print('切分完成！')
    print('切分后列表:', str(len(cut_list)))
    data_cut = list(set(map(lambda i: tuple(i), cut_list)))

    print('去重后列表长度:', str(len(data_cut)))
    
    min_list, max_list = [], []

    for key in data_cut:
        if key[1] <= 2048 and key[1] >= 2000:
            min_list.append(key[0])
        else:
            max_list.append(key)
    # min_list = [k[0] for k in data_cut if  k[1] <= 2048 and k[1] >= 2000]
    # max_list = [h for h in data_cut if h[0] not in min_list]
    print('过滤后:', str(len(max_list)))
    data_chunk = get_chunk(max_list, 2048)
    data_chunk += min_list
    return data_chunk


def get_file(data, file_name):
    pre_data =[line.replace('  ', '').replace('\u2002','').replace('\u3000','').replace('\t','')  for line in  data]

    print(len(pre_data), len(''.join(pre_data)))
    data_chunk = get_pretrain_text(pre_data)

    print(len(data_chunk), len(''.join(data_chunk)))
    cut_dataset = []
    max_tok = []

    for line in data_chunk:
        k = tokenizer_llama.encode(line)
        max_tok.append(len(k))
        if len(k) <= 2048:
            cut_dataset.append(line)
            json_str = json.dumps({'text':line}, ensure_ascii=False)
            with open('./pretrain_data_llama/'+file_name, 'a+', encoding='utf-8') as file_w:
                file_w.write(json_str + '\n')

    print(len(cut_dataset), len(''.join(cut_dataset)))
    print(max(max_tok))

'''
books
'''
b_text = []
b_path = './new_data/books/'
for bt in os.listdir(b_path):
    print(bt)
    b_text += json.load(open(b_path+bt, 'r', encoding='utf-8'))

print(len(b_text))
get_file(b_text, 'books.json')

'''
llama或者bloom都可以使用，替换tokenizer即可
按照token长度进行拼接

首先进行文本token长度进行长度切分，按照2048的倍数+1进行切分，切分后文本和长度存入列表，此步骤使用32进程进行操作；
进行去重；
进行拼接【发现32进行速度竟然很慢，估计什么bug没发现】
'''
