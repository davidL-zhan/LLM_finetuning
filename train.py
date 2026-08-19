from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
from process_data import get_dataloader
from config import ProjectConfig
import torch
from torch.amp import autocast
import time
import os
from utils import CastOutputToFloat, save_model, second2time

config = ProjectConfig()


def evaluate_model(model, dev_dataloader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if config.use_lora:
                with autocast(device_type=config.device, dtype=torch.bfloat16):
                    loss = model(
                        input_ids=batch["input_ids"].to(
                            dtype=torch.long, device=config.device
                        ),
                        labels=batch["labels"].to(
                            dtype=torch.long, device=config.device
                        ),
                    ).loss
            else:
                loss = model(
                    input_ids=batch["input_ids"].to(
                        dtype=torch.long, device=config.device
                    ),
                    labels=batch["labels"].to(dtype=torch.long, device=config.device),
                ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


def train():
    model_config = AutoConfig.from_pretrained(config.pre_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.pre_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.pre_model,
        dtype=torch.bfloat16,
        config=model_config,
    )
    # # 梯度检查点是一种优化技术，用于在反向传播过程中降低内存使用
    # # 保存部分激活值，未保存的反向传播时重新计算
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # 不进行缓存，减少内存
    model.config.use_cache = False

    if config.use_lora:
        model.lm_head = CastOutputToFloat(model.lm_head)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # 推理时为True，比如决定是否使用dropout
            r=config.lora_rank,  # 低秩矩阵维度
            lora_alpha=32,  # 缩放系数
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)

    model = model.to(torch.device(config.device))
    model.print_trainable_parameters()
    print("Model type:", type(model))
    print(model)

    # params = [n for n, p in model.named_parameters()]
    # print(params)
    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(config.train_path),
            "dev": str(config.dev_path),
        },
    )
    train_loader = get_dataloader(dataset["train"], tokenizer,batch_size=config.batch_size)
    dev_loader = get_dataloader(dataset["dev"], tokenizer,batch_size=config.batch_size)

    # # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_loader)
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = config.epochs * num_update_steps_per_epoch
    warm_steps = int(config.warmup_ratio * max_train_steps)  # 预热阶段的训练步数
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    # # 定义训练的一些参数变量
    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float("inf")
    for epoch in range(1, config.epochs + 1):
        print("开始训练")
        for batch in train_loader:
            if config.use_lora:
                # torch.cuda.amp.autocast是PyTorch中一种混合精度的技术（仅在GPU上训练时可使用）
                with autocast(device_type=config.device, dtype=torch.bfloat16):
                    loss = model.forward(
                        input_ids=batch["input_ids"].to(
                            dtype=torch.long, device=config.device
                        ),
                        labels=batch["labels"].to(
                            dtype=torch.long, device=config.device
                        ),
                    ).loss
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            lr_scheduler.step()
            #
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % config.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print(
                    "global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s"
                    % (
                        global_step,
                        global_step / max_train_steps * 100,
                        epoch,
                        loss_avg,
                        config.logging_steps / time_diff,
                        second2time(
                            int(
                                (max_train_steps - global_step)
                                / (config.logging_steps / time_diff)
                            )
                        ),
                    )
                )
                tic_train = time.time()
            if global_step % config.save_freq == 0:
                # cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
                # save_model(model, cur_save_dir)
                # tokenizer.save_pretrained(cur_save_dir)
                # print(f'Model has saved at {cur_save_dir}.')

                eval_loss = evaluate_model(model, dev_loader)

                print("Evaluation Loss: %.5f" % (eval_loss))
                if eval_loss < best_eval_loss:
                    print(
                        f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}"
                    )
                    best_eval_loss = eval_loss
                    cur_save_dir = os.path.join(config.save_dir, "model_best")
                    save_model(model, cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                    print(f"Best model has saved at {cur_save_dir}.")
                tic_train = time.time()


if __name__ == "__main__":
    train()
