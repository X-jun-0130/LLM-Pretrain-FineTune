from sentence_transformers import util
import numpy as np


def get_deduplication(alpha, model, th=0.9):

    batch_size = 512 
    #pool = model.start_multi_process_pool()
    # batches = [alpha[i:i + batch_size] for i in range(0, len(alpha), batch_size)]
    # embedding = model.encode_multi_process(alpha, pool, batch_size=batch_size)
    embedding = model.encode(alpha, batch_size=batch_size)

    # embedding = model.encode([k for k in alpha])
    print("embedding_success")

    # 计算相似度矩阵
    similarity_matrix = util.cos_sim(embedding, embedding)
    print("similarity_matrix_success")

    # 找到相似度大于0.9的元素，并记录要删除的索引
    to_delete = set()
    threshold = th
    rows, cols = np.where(similarity_matrix > threshold)

    filtered_pairs = [(r, c) for r, c in zip(rows, cols) if r < c]

    for r, c in filtered_pairs:
        to_delete.add(c)

    print("to_delete", str(len(to_delete)))
    # 删除相似的元素
    #alpha = [item for idx, item in enumerate(alpha) if idx not in to_delete]
    return to_delete




# from sentence_transformers import SentenceTransformer
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# model = SentenceTransformer('/data/public/text2vec-large-chinese/')
# import json

# cp_list = [json.loads(k) for k in open('./med_sft/report/instruction_single_report.jsonl', 'r', encoding='utf-8')]

# input_kg = [k['instruction'] for k in cp_list]  
# to_delete = get_deduplication(input_kg, model, 0.95)
# _coig = [item for idx, item in enumerate(cp_list) if idx not in to_delete]

# with open('./med_sft/report/instruction_single_report_v2.jsonl', 'a', encoding='utf-8') as target_file:
#     for item in _coig:
#         target_file.write(json.dumps(item, ensure_ascii=False) + '\n')
