# LLM-FineTune
Deepspeed、Bloom

#### Model
    Bloom_7B1
 
#### Deepspeed
    zero_3、cpuoffload、fp16
 
#### Gpus
    4*48G A6000

#### Para
    length<500,batchsize=32
   
#### Requriements
    pytorch=1.13.1  deepspeed=0.7.5  tansformers=4.21.0

#### Projects
    1.FineTune: deepspeed --master_addr 0.0.0.0 --master_port 6006 --include localhost:0,1,2,3 ./Model_Bloom.py
    2.convert_deepspeedmodel_fp32: python model_convert32_save.py
    3.inference:python test.py