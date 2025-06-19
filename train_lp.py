import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

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


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torchvision.models as models

# 假设你已经定义了以下内容：
# from data.imagenet import HuggingfaceImageNet # 或者你自己的 ImageNet 数据集
# from utils.logger import setup_logger # 或者你自己的 logger
# from accelerate import Accelerator # 或者你自己的 Accelerator
# from omegaconf import OmegaConf  # 如果你需要使用配置

# ------------------- Model Definition -------------------
class FeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, num_classes=1000):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            # Remove the classification head (fc layer)
            self.model = nn.Sequential(*list(self.model.children())[:-1]) # remove last layer
            self.feature_size = 2048
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_size = 512
        elif model_name == "vit_b_16":
            self.model = models.vit_b_16(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_size = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        for param in self.model.parameters():
            param.requires_grad = False # Freeze the feature extractor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # for resnet
        self.flatten = nn.Flatten() # flatten layer
        self.linear_probe = nn.Linear(self.feature_size, num_classes) # Linear probe

    def forward(self, x):
        with torch.no_grad(): # No need to compute gradients for the feature extractor
            x = self.model(x)
            x = self.avgpool(x)
            x = self.flatten(x)
        x = self.linear_probe(x) # Apply the linear probe
        return x

# ------------------- Training Functions -------------------
def train_one_epoch(model, train_dataloader, optimizer, criterion, device, logger, accelerator, lr_scheduler=None):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels, _) in enumerate(tqdm(train_dataloader, desc="Training", disable=not accelerator.is_main_process)): # Assuming data loader returns (images, labels)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Use accelerator to handle backward and optimizer step
        accelerator.backward(loss) # Replaces loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0 and accelerator.is_main_process:
            logger.info(f"Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataloader)
    logger.info(f"Epoch Loss: {epoch_loss:.4f}")
    return epoch_loss

def evaluate(model, val_dataloader, device, logger, accelerator):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(val_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# ------------------- Main Training Function -------------------
def main():
    # --- Configuration ---
    # Assuming you have a config or can set these directly
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 64
    num_workers = 8
    log_level = "INFO"
    output_dir = "output"
    model_name = "resnet50" # Choose a model, e.g., "resnet50", "resnet18", "vit_b_16"

    # --- Setup ---
    logger = setup_logger(name="ImageNet-Training", log_level=log_level)
    accelerator = Accelerator()
    device = accelerator.device  # Get device from accelerator

    # Use your dataset and dataloader
    train_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "train")  # Replace with your dataset
    val_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "validation")  # Replace with your dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    # --- Model, Optimizer, Loss, and Scheduler ---
    model = FeatureExtractor(model_name=model_name)
    model = model.to(device) # Move the model to the correct device
    optimizer = optim.Adam(model.linear_probe.parameters(), lr=learning_rate)  # Only optimize the linear probe
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = None #  You can add a learning rate scheduler if needed

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, val_dataloader,  = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # --- Training Loop ---
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device, logger, accelerator, lr_scheduler
        )
        accuracy = evaluate(model, val_dataloader, device, logger, accelerator)

        # Save the best model (optional)
        if accuracy > best_accuracy and accelerator.is_main_process:
            best_accuracy = accuracy
            accelerator.wait_for_everyone()  # Ensure all processes are synchronized before saving
            unwrapped_model = accelerator.unwrap_model(model) # unwrap
            torch.save(unwrapped_model.linear_probe.state_dict(), os.path.join(output_dir, "best_model.pth")) # save the linear layer only
            logger.info(f"Saved best model with accuracy: {best_accuracy:.2f}%")

    logger.info("Training finished.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
