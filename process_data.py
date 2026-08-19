from datasets import load_dataset
from config import config


def convert_example(
    example,
    tokenizer,
):
    return {
        "prompt": [
            {
                "role": "user",
                "content": example["context"],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["target"],
            }
        ],
    }


# print("converted sample:")
# print(train_dataset[0])
# # print(convert_example(dataset["train"][0]))


def get_dataloader(
    dataset,
    split="train",
    shuffle=True,
    batch_size=4,
):
    dataset = dataset.map(
        convert_example,
        remove_columns=eval_dataset.column_names,
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
    train_loader = get_dataloader(train_dataset)
