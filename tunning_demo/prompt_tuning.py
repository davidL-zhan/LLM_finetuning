from peft import get_peft_model, PromptTuningConfig, TaskType
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "google-bert/bert-base-chinese"


tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init="RANDOM",
    num_virtual_tokens=20,  # 虚拟词数
    tokenizer_name_or_path=model_name,
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()


input_text = "今天天气不错"
input_label = 1
inputs = tokenizer(
    input_text, return_tensors="pt", padding=True, truncation=True, max_length=128
)
print(inputs)
inputs["labels"] = torch.tensor([input_label])

optimizer = torch.optim.AdamW(peft_model.prompt_encoder.parameters(), lr=1e-5)
outputs = peft_model(**inputs)
loss = outputs.loss
loss.backward()
optimizer.step()

print(outputs)
logits = outputs.logits
pred = logits.argmax(-1)
print(f"预测结果：{pred}")
