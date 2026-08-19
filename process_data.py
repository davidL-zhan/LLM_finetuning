from datasets import load_dataset
from config import config
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from torch.utils.data import DataLoader


def convert_data(
    example,
):

    return [
        {
            "role": "user",
            "content": example["context"],
        },
        {
            "role": "assistant",
            "content": example["target"],
        },
    ]


def tokenize_example(example, tokenizer, max_source_len, max_target_len):
    messages = convert_data(example)
    # print(f"messages-----{messages}")
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    # print(f"messages[:-1]-----{messages[:-1]}")
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids

    labels = full_ids.copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    return {"input_ids": full_ids, "labels": labels}


def get_dataloader(dataset, tokenizer, shuffle=True, batch_size=4, num_workers=0):

    dataset = dataset.map(
        tokenize_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_len": config.max_source_seq_len,  # 100
            "max_target_len": config.max_target_seq_len,
        },
        remove_columns=dataset.column_names,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,  # Windows 下多进程 dataloader 会报错，必须为 0
        pin_memory=True,  # 可选：GPU 训练时加速数据搬运
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.pre_model)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.train_path),
            "dev": str(config.dev_path),
        },
    )
    print(dataset)
    print(dataset["train"][0])
    train_dataset, eval_dataset = dataset["train"], dataset["dev"]
    # train_loader = get_dataloader(train_dataset)
    train_loader = get_dataloader(
        train_dataset, tokenizer, shuffle=True, batch_size=config.batch_size
    )
    # 验证：检查一个 batch 的形状和 label 部分内容
    batch = next(iter(train_loader))
    # print(batch)
    print({k: v.shape for k, v in batch.items()})  # input_ids / attention_mask / labels
    print(tokenizer.decode(batch["input_ids"][0][batch["labels"][0] != -100]))
