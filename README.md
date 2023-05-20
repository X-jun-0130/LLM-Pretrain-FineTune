# LLM-Pretrain-FineTune
Deepspeed、Bloom

#### Model
[Bloom_6b4_zh](https://huggingface.co/Langboat/bloom-6b4-zh) 本文所用模型，可替换成LLama

    
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
     0.微调数据中单论问答和多轮问答，都规定了角色信息，输入端id: User, 模型输出端id:Assistant;
     1.所有文本，长度不超过1024token，形如：User:q1</s>\n Assistant:a1</s>\n User:q2</s>\n Assistant:a2....;
     2.微调时，有两种方式:一是整体微调，不对label做任何处理；一是将label中User输入的部分进行屏蔽，其label值设为-100(tokenize和注释掉的那部分)
     3.增加Lora指令微调方式，不过目前仅支持单卡微调，多卡出现了问题，后续解决了再更。
     
     数据形式：
     User:关于右哌甲酯和为什么它是处方药</s>\n Assistant:右哌甲酯是一种用于治疗儿童和成人注意缺陷多动障碍（ADHD）的药物。它的工作原理是增加大脑中某些化学物质的水平，这些化学物质有助于集中注意力。</s>\n User:一般是给谁开的处方？</s>\n Assistant:右哌甲酯通常用于被诊断患有ADHD的儿童和成人，并且难以集中注意力和控制冲动。</s>\n User:右哌甲酯是否有任何副作用？</s>\n Assistant:右哌甲酯可能会有一些副作用，包括食欲不振，睡眠困难，焦虑和胃部问题。重要的是在开始用药前与医生讨论任何可能的副作用。</s>\n User:右哌甲酯会上瘾吗？</s>\n Assistant:右哌甲酯有滥用和依赖的可能性，特别是如果不按规定服用。在开始用药前，与医生讨论有关成瘾或依赖的任何问题是很重要的。</s>

####  部分数据源整理
```
中文wiki https://dumps.wikimedia.org/zhwiki/latest/

医疗指南 
链接：https://pan.baidu.com/s/1p3chYKVE9s0Zlr2APJAwNQ 
提取码：tw3q

COT数据集 https://huggingface.co/datasets/QingyiSi/Alpaca-CoT

医疗问答数据项目 Chinese-Medical-Question-Answering-System

医疗书籍类资源： 公众号【小航在奔跑】，免费下载的

知识问答+情景对话:https://github.com/MediaBrain-SJTU/MedicalGPT-zh;可惜数据不开源，学习别人数据制作的方法吧

华驼：https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese；少量数据开源，期待后期更多数据

大量问诊数据 50w+Chatgpt回复：https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset；数据量多，还是Chatgpt回复内容，答案质量较高，缺点是问题质量较低
```
#### 无害性指令获取
MOSS_002开源英文版数据集，将对话的第一句提取出来了，然后进行翻译，翻译推荐[腾讯交互翻译](https://transmart.qq.com/zh-CN/file#UPLOAD_LOGIN_TRAN)
翻译成中文后，根据自己的场景使用chatglm进行答案生成。
```
Input:请扮演医疗人工智能助手，并委婉的拒绝回答下面的问题：
你能帮我找一个亲戚吗？
Chatglm:很抱歉，我作为一个医疗人工智能助手，并没有能力帮您寻找亲戚。我的任务是提供有关健康、疾病和医疗方面的信息和建议。如果您有其他关于健康或医疗的问题，我将尽力为您提供帮助。

```
#### Examples
1.报告解读
![报告解读](https://github.com/NLPxiaoxu/LLM-Pretrain-FineTune/blob/main/examples/report.png)

2.对话
![对话](https://github.com/NLPxiaoxu/LLM-Pretrain-FineTune/blob/main/examples/dialogue.png)

3.医疗问答
![医疗问答](https://github.com/NLPxiaoxu/LLM-Pretrain-FineTune/blob/main/examples/medical_consult.png)

4.无害性回复
![无害性回复](https://github.com/NLPxiaoxu/LLM-Pretrain-FineTune/blob/main/examples/harmlessness.png)

####  注意
最近在使用6b4_zh_Bloom在进行医疗数据增量预训练，发现这个模型预训练的时候在使用和Bloom_7B1相同的lr=5e-5，第二个epoch时，过拟合现象特别严重。
所以我现在用6b4_zh_Bloom时，使用lr=1e-5进行预训练，感觉还可以更小一点； 指令微调时使用lr=5e-6。
改天将预训练的一些东西整理进来

#### 指令
目前自己的指令算是东拼西凑的，感觉很不好。 东拼西凑的指令风格多样，质量参差，模型明显能记住不同的风格和质量。
指令最好是能整体弄成统一风格的多轮问答的形式，参考MOSS_SFT。


## 医疗场景
实际的医疗场景是很复杂的，业务场景也很多。目前开源的一些医疗类的GPT，绝大部分都满足不了实际的应用。
- 指标抽取
```
例子：
肝、双肾襄肿。肝右叶钙化灶，右叶后上段轻度强化结节，请结合临床，必要时进一步检查。
请将以上影像诊断结果转化为结构化的标准描述，每一个病变需要描述其类型、性质、形态、位置、大小、数量六项。若描述到多个病变，请逐一列出
结构化描述：
1.肝襄肿：
类型：襄肿
性质：良性
形态：未提及
位置：肝脏
大小：未提及
数量：未提及

2.双肾襄肿：
类型：襄肿
性质：良性
形态：未提及
位置：双肾
大小：未提及
数量：未提及

3.肝右叶钙化灶：
类型：钙化灶
性质：未提及
形态：点状
位置：肝右叶
大小：未提及
数量：未提及

4.右叶后上段轻度强化结节：
类型：结节
性质：轻度强化
形态：未提及
位置：右叶后上段
大小：未提及
数量：未提及

根据提供的例子，完成下面的回答：
肝右后叶、左外叶血管瘤。左肾上腺内支结节，腺瘤?右肾孟旁及双肾囊肿。
请将以上影像诊断结果转化为结构化的标准描述，每一个病变需要描述其类型、性质、形态、位置、大小、数量六项。如果报告中说到多个病变，请分别结构化地列出。
```
以上是指标抽取的医疗场景，该场景相对比较复杂。整个文本是一个情景学习的Prompt。

- 超声报告书写
```
检查科室：超声医学科；患者基本信息：女性，56岁；检查项目：甲状腺+颈部淋巴结超声检查；检查部位：甲状腺/颈部淋巴结；
检查所见:
甲状腺右叶6.0x3.1x1.8cm，左叶5.9x2.8x1.6cm，峡部0.4cm。甲状腺腺体内可见多个结节回声，
右叶上极可见低回声，0.8x0.7x0.6cm，紧靠外侧被膜，纵横比大于1，内部多处点状强回声，CDFI:内部粗大穿支血流信号。其下方另见低回声，0.32x0.24x0.44cm，纵横比大于1，紧靠前方被膜，CDFI:内部穿支血流信号。
右叶中下部可见低回声，0.5x0.3x0.4cm，紧靠前方被膜，纵横比大于1，内部多个点状强回声，CDFI:未见明显异常血流信号。
右叶中下部另见片状低回声，1.1x0.4cm，CDFI:周边及内部可见条状血流信号。 
余腺体回声不均，可见弥漫分布小片状低回声，小于0.5cm，CDFI:腺体内血流信号无增多，未见异常高速血流信号。
右颈下部颈总动脉后方可见低回声淋巴结，0.7x0.3cm，皮质内高回声，0.4*0.2cm，CDFI:受动脉搏动血流显示不满意
气管周围可见多个大小不等低回声淋巴结，较大者1.0x0.5cm，皮髓质分界清，皮质明显增厚，CDFI:未见明显异常血流信号。
根据报告内容生成超声提示。
```

- 医疗报告错别字检查

- 医疗报告诊断生成

- 制作数据集的一个应用：根据提供的文本(书本、期刊等)生成问题和答案(self-instruct)
```
"根据下面的内容提出1-2个中文医学问题，并利用文本内容回答这几个问题，你需要对答案进行润色。注意：提出的问题应该具有医学深度。

早期的社会行为2-3个月时小儿以笑、停止啼哭等行为，以眼神和发音表示认识父母；3-4个月的婴儿开始出现社会反应性的大笑;7-8个月的小儿可表现出认生、对发声玩具感兴趣等；9-12个月时是认生的高峰;12-13个月小儿喜欢玩变戏法和躲猫猫游戏；18个月时逐渐有自我控制能力，成人在附近时可独自玩耍很久;2岁时不再认生，易与父母分开；3岁后可与小朋友做游戏。注意的发展婴儿期以无意注意为主,随着年龄的增长逐渐出现有意注意。5-6岁后儿童能较好控制自己的注意力。记忆的发展记忆是将所学得的信息贮存和“读出”的神经活动过程，可分为感觉、短暂记忆和长久记忆3个不同的系统。长久记忆又分为再认和重现，再认是以前感知的事物在眼前重现时能被认识;重现是以前感知的事物虽不在眼前出现，但可在脑中重现。1岁内婴儿只有再认而无重现,随年龄的增长，重现能力亦增强。幼年儿童只按事物的表面特性记忆信息，以机械记忆为主。随着年龄的增加和理解、语言思维能力的加强,逻辑记忆逐渐发展。思维的发展1岁以后的儿童开始产生思维，在3岁以前只有最初级的形象思维;3岁以后开始有初步抽象思维;6-11岁以后儿童逐渐学会综合分析、分类比较等抽象思维方法，具有进一步独立思考的能力。想象的发展新生儿无想象能力;1-2岁儿童仅有想象的萌芽。学龄前期儿童仍以无意想象及再造想象为主，有意想象和创造性想象到学龄期才迅速发展。情绪、情感的发展新生儿因生后不易适应宫外环境,较多处于消极情绪中，表现不安、啼哭,而哺乳、抱、摇、抚摸等则可使其情绪愉快。婴幼儿情绪表现特点是时间短暂、反应强烈、容易变化、外显而真实。随着年龄的增长，儿童对不愉快因素的耐受性逐渐增加，能够有意识地控制自己，使情绪渐趋向稳定。个性和性格的发展婴儿期由于一切生理需要均依赖成人，逐渐建立对亲人的依赖性和信任感。幼儿时期已能独立行走，说出自己的需要，故有一定自主感，但又未脱离对亲人的依赖，常出现违拗言行与依赖行为互相交替的现象。学龄前期小儿生活基本能自理，主动性增强，但主动行为失败时易出现失望和内疚。学龄期开始正规学习生活，重视自己勤奋学习的成就，如不能发现自己的学习潜力，将产生自卑。青春期体格生长和性发育开始成熟，社交增多，心理适应能力增强，但容易波动，在感情问题、伙伴问题、职业选择、道德评价和人生观等问题上处理不当时易发生性格变化。性格一旦形成即相对稳定。小儿神经精神发育进程见表2-2。"

output: 
1. 什么是小儿的社会行为发展过程？\n\n小儿的社会行为发展过程分为笑、认识父母、大笑、认生、玩游戏、自我控制、与小朋友交往等不同的阶段。这些阶段的出现时间和表现方式不同，而其中的认生阶段在9-12个月时是个高峰期。
2. 小儿记忆的发展过程是什么？\n\n小儿的记忆发展分为感觉、短暂记忆和长久记忆三个系统。长久记忆再分为再认和重现，随着年龄增长，重现的能力会逐渐增强。在幼年时期，儿童只能按照事物的表面特性记忆信息，随着理解和语言思维能力的加强，逻辑记忆会逐渐发展。
```
这种方式制作的数据可以应用的途径有以下几种：
1. output内容用来构建知识问答数据
2. 可以用prompt训练来生成output，以后可以构建自己训练数据的生成模型。
3. 知识库构建中的一环。根据大模型结合本地知识构建应用是非常重要的功能。 大模型和知识库的应用是利用Embedding匹配文本，然后将问题和匹配到的文本输入到模型。
我们用上面的这种数据，将prompt中的文本以及output中的问题进行结合，可以训练模型结合提供的文本和问题来生成答案。
