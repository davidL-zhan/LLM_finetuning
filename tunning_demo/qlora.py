import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

# 1. 加载预训练模型和分词器
# 选择一个基础模型。实际QLoRA通常用于几十亿到几千亿参数的模型，
# 这里为了演示方便和在消费级GPU上运行，我们使用一个较小的模型。
model_name_or_path = "google-bert/bert-base-chinese"  # 仅为演示使用的小模型

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 对于因果语言模型，通常需要设置 pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 配置 4-bit 量化和 LoRA
# BitsAndBytesConfig 用于配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4-bit 量化
    bnb_4bit_quant_type="nf4",  # 使用 NormalFloat4 (NF4) 量化类型
    bnb_4bit_use_double_quant=True,  # 启用双量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算数据类型，推荐 bfloat16 (如果GPU支持) 或 float16
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,  # LoRA 的秩
    lora_alpha=32,  # LoRA 缩放因子 (通常为 r 的两倍)
    lora_dropout=0.1,  # LoRA 层 dropout
    bias="none",  # 不对 bias 层应用 LoRA
    task_type=TaskType.SEQ_CLS,  # 任务类型：因果语言模型 (生成)
)

# 3. 加载并量化基础模型
# device_map="auto" 让 accelerate 自动将模型加载到可用的设备上，并处理多 GPU 分布
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,  # 将量化配置传递给模型加载函数
    dtype=torch.bfloat16,  # 同样设置模型的 dtype
)

# 为 4-bit 训练准备模型：这会处理一些必要的设置，例如将 lm_head 设置为 float32
model = prepare_model_for_kbit_training(model)
print("model--->", model)


# 应用 LoRA 配置到量化后的模型
model = get_peft_model(model, lora_config)

# 打印模型的可训练参数，你会发现参数量非常小，且只包含 LoRA 相关的参数
print(f"\nModel trainable parameters (QLoRA):")
model.print_trainable_parameters()
