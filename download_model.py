import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Model type:", type(model))
print("Model device:", model.device)
print("Model dtype:", model.dtype)
print("Number of parameters:", model.num_parameters())
input()
