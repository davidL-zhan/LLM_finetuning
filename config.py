# -*- coding:utf-8 -*-
import torch
import os
from pathlib import Path


class ProjectConfig(object):
    def __init__(self):
        self.cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
        print(self.cur_path)
        # 定义是否使用GPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 定义Qwen3-4B模型的路径的名称
        self.pre_model = "Qwen/Qwen3-4B-Instruct-2507"
        # 定义训练数据的路径
        self.train_path = self.cur_path / "data\mixed_train_dataset.jsonl"
        # 定义验证集的路径
        self.dev_path = self.cur_path / "data\mixed_dev_dataset.jsonl"
        # 是否使用LoRA方法微调
        self.use_lora = True
        # 是否使用P-Tuing方法微调
        self.use_ptuning = False
        # 秩==8
        self.lora_rank = 8
        # 一个批次多少样本
        self.batch_size = 4
        # 训练几轮
        self.epochs = 2
        # 学习率
        self.learning_rate = 3e-5
        # 权重权重系数
        self.weight_decay = 0
        # 学习率预热比例
        self.warmup_ratio = 0.06
        # context文本的输入长度限制
        self.max_source_seq_len = 100
        # target文本长度限制
        self.max_target_seq_len = 100
        # 每隔多少步打印日志
        self.logging_steps = 10
        # 每隔多少步保存
        self.save_freq = 200
        # 如果你使用了P-Tuing，要定义伪tokens的长度
        self.pre_seq_len = 200
        self.prefix_projection = (
            False  # 默认为False,即p-tuning,如果为True，即p-tuning-v2
        )
        # 保存模型的路径
        self.save_dir = "/gemini/checkpoints/ptune"


config = ProjectConfig()


if __name__ == "__main__":
    ...
