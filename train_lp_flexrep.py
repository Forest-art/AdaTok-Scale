"""Training script for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torchvision.models as models

import torch
from omegaconf import OmegaConf
from data.imagenet import ImageNetDataset, HuggingfaceImageNet
from utils.logger import setup_logger
from torch.utils.data import Dataset, DataLoader
from utils.base_utils import instantiate_from_config
from utils.train_utils_flextok import (
    get_config, create_pretrained_tokenizer, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)
import warnings
warnings.filterwarnings("ignore")
from PIL import PngImagePlugin

import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Assuming these imports are from your project
from utils.train_utils_flextok import train_one_epoch as original_train_one_epoch
from utils.train_utils_flextok import save_checkpoint
from utils.logger import setup_logger

class LinProb(nn.Module):
    def __init__(self, in_dim=8, num_classes=1000):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.bn = torch.nn.BatchNorm1d(self.in_dim, affine=False, eps=1e-6)
        self.head = torch.nn.Linear(self.in_dim, self.num_classes)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(self.head.weight, std=0.01)
    
    def forward(self, z):
        return self.head(self.bn(z))

    
def evaluate(model, val_dataloader, device, logger, accelerator, linear_probe):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model.module.encoder(images, model.module.latent_tokens)[1]
            keep_lengths = torch.randint(31, 32, (outputs.shape[0],))

            mean_tokens = []

            for i in range(images.shape[0]):
                # 获取当前样本前 keep_lengths[i] 个 token
                keep_len = keep_lengths[i]
                selected_tokens = outputs[i, :keep_len, :]  # 选择前 keep_len 个 token
                mean_tokens.append(selected_tokens.mean(dim=0))  # 沿着 token 维度求均值

            # 将 mean_tokens 转换为 tensor，形状为 (batch_size, 512)
            mean_tokens = torch.stack(mean_tokens)
            outputs = linear_probe(mean_tokens)
            
            # Top-1 Accuracy
            _, top1_predicted = torch.max(outputs.data, 1)
            top1_correct += (top1_predicted == labels).sum().item()
            
            # Top-5 Accuracy
            _, top5_predicted = torch.topk(outputs.data, 5, dim=1)
            top5_correct += torch.sum(top5_predicted == labels.view(-1, 1)).float().sum().item()

            total += labels.size(0)

    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total

    logger.info(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    logger.info(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    return top1_accuracy, top5_accuracy


def train_one_epoch(model, train_dataloader, optimizer, criterion, device, logger, accelerator, lr_scheduler=None, linear_probe=None):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels, clip_images) in enumerate(tqdm(train_dataloader, desc="Training", disable=not accelerator.is_main_process)): # Assuming data loader returns (images, labels)
        images, labels, clip_images = images.to(device), labels.to(device), clip_images.to(device)

        optimizer.zero_grad()
        outputs = model.module.encoder(images, model.module.latent_tokens)[1]
        keep_lengths = torch.randint(1, model.module.num_latent_tokens, (outputs.shape[0],))

        mean_tokens = []

        for i in range(images.shape[0]):
            # 获取当前样本前 keep_lengths[i] 个 token
            keep_len = keep_lengths[i]
            selected_tokens = outputs[i, :keep_len, :]  # 选择前 keep_len 个 token
            mean_tokens.append(selected_tokens.mean(dim=0))  # 沿着 token 维度求均值

        # 将 mean_tokens 转换为 tensor，形状为 (batch_size, 512)
        mean_tokens = torch.stack(mean_tokens)
        outputs = linear_probe(mean_tokens)
        loss = criterion(outputs, labels)

        # Use accelerator to handle backward and optimizer step
        accelerator.backward(loss) # Replaces loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0 and accelerator.is_main_process:
            logger.info(f"Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataloader)
    logger.info(f"Epoch Loss: {epoch_loss:.4f}")
    return epoch_loss


# === Main Training Loop ===
def main():
    # Configurations and setup
    config = get_config()  # Assuming your get_config function is defined elsewhere
    logger = setup_logger(name="LinearProbe-Logger", log_level="INFO")
    accelerator = Accelerator()  # Assuming Accelerator is correctly initialized

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # # Initialize model, optimizer, dataloaders, etc.
    model, ema_model, loss_module = create_model_and_loss_module(config, logger, accelerator, model_type="flexrep-tok")
    model = model.cuda()
    
    model.load_state_dict(torch.load("/project/peilab/luxiaocheng/projects/FlexRep/pretrained_models/flexrep_s256_stage1_1000k_randmask.bin"), strict=True)


    # # Freeze all layers except for the final layer
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze all layers


    # model = FeatureExtractor(model_name="resnet50")
    model = model.to(accelerator.device) # Move the model to the correct device

    linear_probe = LinProb(512, 1000)
    linear_probe = linear_probe.cuda()

    linear_probe.load_state_dict(torch.load("/project/peilab/luxiaocheng/projects/FlexRep/exp/FlexRep_s256_tok_stage1/linear_probe_epoch_20.pth"), strict=True)

    criterion = nn.CrossEntropyLoss()

    import torch.optim as optim
    optimizer = optim.Adam(linear_probe.parameters(), lr=0.001)
    lr_scheduler = None #  You can add a learning rate scheduler if needed

    train_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "train")
    val_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "validation")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, val_dataloader,  = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # --- Training Loop ---
    best_accuracy = 0.0
    for epoch in range(6, 20):
        logger.info(f"Epoch {epoch+1}/{20}")
        # train_loss = train_one_epoch(
        #     model, train_dataloader, optimizer, criterion, accelerator.device, logger, accelerator, lr_scheduler, linear_probe
        # )
        accuracy = evaluate(model, val_dataloader, accelerator.device, logger, accelerator, linear_probe)

        save_probe_checkpoint(linear_probe, epoch, output_dir)

    logger.info("Training finished.")
    accelerator.end_training()

# Function to save the probe model after each epoch
def save_probe_checkpoint(model, epoch, output_dir, prefix="linear_probe"):
    checkpoint_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved probe model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
