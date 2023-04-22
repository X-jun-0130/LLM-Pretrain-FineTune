# LLM-Pretrain-FineTune
Deepspeed、Bloom

#### Model
    Bloom_7B1
 
#### Deepspeed
    zero_3、cpuoffload、fp16
 
#### Gpus
    8*48G A6000

#### Para
    token=1024,batchsize=32*8
    token=2048,batchsize=8*8
   
#### Requriements
    pytorch=1.13.1  deepspeed=0.7.5  tansformers=4.21.0
    
    transformers=4.28.1带有大模型的生成效果【流式输出】，最近进行了升级，同时deepspeed升级为0.8.3，torch没变

#### Projects
    0.Pretrain: deepspeed --master_addr 0.0.0.0 --master_port 6006 --include localhost:0,1,2,3,4,5,6,7 ./Model_Bloom_Pretrain.py
    1.FineTune: deepspeed --master_addr 0.0.0.0 --master_port 6006 --include localhost:0,1,2,3,4,5,6,7 ./Model_Bloom_Sft.py
    2.convert_deepspeedmodel_fp32: python model_convert32_save.py
    3.inference: python test.py
    4.api: python Bloom_api.py
    
  
#### gradient_checkpointing
    use_cache=False;  batch_size 可以增大10倍以上
    token =1024,batchsize=32*8

#### 预训练数据处理
     0.书籍这类数据章节内容都超长，使用了滑动窗口取数的方式，1024token的，书籍窗口设置1100，步长设置950，句子有150字符是重复，为了模型能够续写记住切断的句子
     1.对于较短的文本，比如问答、选择题这类数据，采用拼接的方式；
     2.预训练的数据没有做什么格式上的处理，比如问答数据：q+a直接组合，中间没有什么特殊符号、多轮问答：保留角色直接组合；
     3.预训练的文本，形如:<s>text</s>形式进行模型
     
#### 微调数据处理
     0.微调数据中单论问答和多轮问答，都规定了角色信息，输入端id: #User, 模型输出端id:#System;
     1.所有文本进行拼接，长度不超过1024token，形如：#User:q1</s>#System:a1</s>#User:q2</s>#System:a2....;
     2.微调时，有两种方式:一是整体微调，不对label做任何处理；一是将label中User输入的部分进行屏蔽，其label值设为-100(tokenize和注释掉的那部分)

####  部分数据源整理
```
中文wiki https://dumps.wikimedia.org/zhwiki/latest/

医疗指南 
链接：https://pan.baidu.com/s/1p3chYKVE9s0Zlr2APJAwNQ 
提取码：tw3q

COT数据集 https://huggingface.co/datasets/QingyiSi/Alpaca-CoT

医疗问答数据项目 Chinese-Medical-Question-Answering-System

医疗书籍类资源： 公众号【小航在奔跑】，免费下载的

MOSS开源了大量指令数据

翻译后的中文Lora数据
```



####  注意
最近在使用6b4_zh_Bloom在进行医疗数据增量预训练，发现这个模型预训练的时候在使用和Bloom_7B1相同的lr=5e-5，第二个epoch时，过拟合现象特别严重。
所以我现在用6b4_zh_Bloom时，使用lr=1e-5进行预训练，感觉还可以更小一点； 指令微调时使用lr=5e-6。
改天将预训练的一些东西整理进来
