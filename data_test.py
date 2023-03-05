import json


qa_list = json.load(open('data/qa_train_data.json', 'r', encoding='utf-8'))
dialogue_list = json.load(open('data/small_dialogue_train_data.json', 'r', encoding='utf-8'))
kg_list = json.load(open('data/kg_drug_train_data.json', 'r', encoding='utf-8'))

qa_dataset = [['<s>' + k['text'] + '</s>' + k['answer'][:500] + '</s>', j] for j, k in enumerate(qa_list[:32])]
dia_dataset = [['<s>' + '</s>'.join(d) + '</s>', i] for i, d in enumerate(dialogue_list[:32])]
kg_dataset = [['<s>' + k['text'] + '</s>' + k['answer'][:500] + '</s>', j] for j, k in enumerate(kg_list[:32])]


print(dia_dataset[0:2])