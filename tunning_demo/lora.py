from peft import LoraConfig, TaskType, get_peft_model
from transformers import BertForSequenceClassification, BertTokenizer

model_name = "google-bert/bert-base-chinese"


tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value", "output.keys"],
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(model)
