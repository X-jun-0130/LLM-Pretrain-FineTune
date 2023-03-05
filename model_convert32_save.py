from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


convert_zero_checkpoint_to_fp32_state_dict('./results/checkpoint-15/', './Bloom_Save/pytorch_model.bin')

