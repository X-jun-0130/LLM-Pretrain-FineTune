# LLM-Pretrain-FineTune
Deepspeed、Bloom

#### Model
    Bloom_7B1
    [Bloom_6b4_zh](https://huggingface.co/Langboat/bloom-6b4-zh) 推荐使用
 
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
     4.注意数据多样性，真的狠重要啊！！！
     
#### 微调数据处理
     0.微调数据中单论问答和多轮问答，都规定了角色信息，输入端id: #User, 模型输出端id:#System;
     1.所有文本进行拼接，长度不超过1024token，形如：#User:q1</s>#System:a1</s>#User:q2</s>#System:a2....;
     2.微调时，有两种方式:一是整体微调，不对label做任何处理；一是将label中User输入的部分进行屏蔽，其label值设为-100(tokenize和注释掉的那部分)
     3.增加Lora指令微调方式，不过目前仅支持单卡微调，多卡出现了问题，后续解决了再更。

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
#### 无害性指令获取
MOSS_002开源英文版数据集，将对话的第一句提取出来了，然后进行翻译，翻译推荐[腾讯交互翻译](https://transmart.qq.com/zh-CN/file#UPLOAD_LOGIN_TRAN)
翻译成中文后，根据自己的场景使用chatglm进行答案生成。
```
Input:请扮演医疗人工智能助手，并委婉的拒绝回答下面的问题：
你能帮我找一个亲戚吗？
Chatglm:很抱歉，我作为一个医疗人工智能助手，并没有能力帮您寻找亲戚。我的任务是提供有关健康、疾病和医疗方面的信息和建议。如果您有其他关于健康或医疗的问题，我将尽力为您提供帮助。

```


####  注意
最近在使用6b4_zh_Bloom在进行医疗数据增量预训练，发现这个模型预训练的时候在使用和Bloom_7B1相同的lr=5e-5，第二个epoch时，过拟合现象特别严重。
所以我现在用6b4_zh_Bloom时，使用lr=1e-5进行预训练，感觉还可以更小一点； 指令微调时使用lr=5e-6。
改天将预训练的一些东西整理进来

#### 指令
目前自己的指令算是东拼西凑的，感觉很不好。 东拼西凑的指令风格多样，质量参差，模型明显能记住不同的风格和质量。
指令最好是能整体弄成统一风格的多轮问答的形式，参考MOSS_SFT。

谁有好的高质量医疗指令获取路径，告知我一下啊！！！
