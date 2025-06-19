import math
import os
from pathlib import Path
from accelerate.utils import set_seed
from accelerate import Accelerator
import inspect
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from utils.logger import setup_logger
from utils.demo_util import get_titok_tokenizer, get_adaptok_tokenizer
from data.imagenet import ImageNetDataset, HuggingfaceImageNet
from utils.train_utils import (
    get_config, create_model_and_loss_module, create_pretrained_tokenizer,
    create_optimizer, create_lr_scheduler, create_dataloader,
    auto_resume, save_checkpoint, train_one_epoch_generator)
from torch.utils.data import Dataset, DataLoader
from utils.base_utils import instantiate_from_config
from modeling.generator.transformer import GPT_models

# LoRA 层的实现
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        """
        :param original_layer: 需要添加 LoRA 的原始层 (例如 nn.Linear)
        :param rank: 低秩矩阵的秩
        :param alpha: 用于缩放 LoRA 参数的系数
        """
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # 获取原始层的输入输出维度
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 低秩矩阵（W的低秩近似）
        self.low_rank_A = nn.Parameter(torch.randn(in_features, rank))  # A矩阵
        self.low_rank_B = nn.Parameter(torch.randn(rank, out_features))  # B矩阵

        # 乘上 alpha 来调整 LoRA 的影响
        self.scaling_factor = alpha / rank

    def forward(self, x):
        # LoRA 的计算： y = xW + ABx
        # 原始层计算
        original_output = self.original_layer(x)
        # LoRA 计算
        lora_output = torch.matmul(torch.matmul(x, self.low_rank_A), self.low_rank_B) * self.scaling_factor
        # 输出：原始输出 + LoRA 适配输出
        return original_output + lora_output

# 自定义的 get_peft_model 函数
def get_peft_model(model, lora_config):
    """
    在模型的指定层上应用 LoRA。
    
    :param model: 原始模型
    :param lora_config: LoRA 配置，包含需要应用 LoRA 的模块
    :return: 应用 LoRA 的模型
    """
    # 遍历模型的每一层
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target_module in name for target_module in lora_config["target_modules"]):
            # 替换原始层为 LoRA 层
            original_layer = module
            # 创建 LoRA 层，并替换原始层
            lora_layer = LoRALayer(original_layer, rank=lora_config["r"], alpha=lora_config["lora_alpha"])
            # 替换原始模型的模块
            parent_module = model
            # 使用 parent_module 递归获取父模块并替换目标层
            parts = name.split('.')
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, parts[-1], lora_layer)
    
    return model


def main():
    workspace = os.environ.get('WORKSPACE', '')
    torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="LlamaGen", log_level="INFO",
                          output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    accelerator.wait_for_everyone()

    tokenizer = get_adaptok_tokenizer(config)
    tokenizer.to(accelerator.device)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="llamagen")
    
    checkpoint = torch.load("./pretrained_models/c2i_B_256.pt", map_location="cpu")
    # Get the state dict from the checkpoint
    checkpoint_state_dict = checkpoint['model']
    # Get the current model's state dict
    model_state_dict = model.state_dict()
    # Filter out keys in the checkpoint that do not match the model's state dict
    filtered_checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
    # Load the filtered state dict into the model
    model.load_state_dict(filtered_checkpoint_state_dict, strict=False)

    learning_rate = 1e-4
    betas = (0.9, 0.999)  # 一般使用 (0.9, 0.999) 默认值
    weight_decay = 1e-5  # 权重衰减，防止过拟合

    # 创建 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    lr_scheduler, _ = create_lr_scheduler(config, logger, accelerator, optimizer)

    train_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "train")
    val_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "validation")
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.per_gpu_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.per_gpu_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0
    global_step, first_epoch = auto_resume(config, logger, accelerator, ema_model, num_update_steps_per_epoch)

    loss_module = torch.nn.CrossEntropyLoss()
    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch_generator(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer,
                            lr_scheduler,
                            train_dataloader,
                            tokenizer,
                            global_step,
                            model_type="llamagen")

        if global_step >= config.training.max_train_steps:
            accelerator.print(f"Finishing training: Global step reached max train steps: {global_step}")
            break

    accelerator.wait_for_everyone()
    save_checkpoint(model, output_dir, accelerator, global_step, logger)
    
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
