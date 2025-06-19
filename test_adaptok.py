import os
import math
from pathlib import Path
import torch
import pprint
import argparse
from datetime import timedelta
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
from utils.train_utils import save_checkpoint, load_checkpoint, create_lr_scheduler, create_evaluator, create_pretrained_tokenizer
from torchmetrics.image.fid import FrechetInceptionDistance
from data.imagenet import ImageNetDataset
from tqdm import tqdm  # 导入 tqdm 库
from modeling.titok import TiTok
import torch.nn.functional as F
from modeling.flextitok import FlexTiTok
from modeling.adaptmask import AdapTok
from utils.base_utils import instantiate_from_config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    pretrained_tokenizer=None
):
    # 评估模式
    num_samples = 0
    total_rMSE = 0
    total_hit = 0
    total_tokens = 0
    model.eval()
    # 初始化 FID 指标（torchmetrics）
    fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
    
    # 确定是否为主进程
    is_main_process = accelerator.is_main_process

    # 初始化进度条，仅在主进程显示
    progress_bar = tqdm(eval_loader, desc="Evaluating Reconstruction", leave=True, disable=not is_main_process)

    for batch in progress_bar:
        # 将批量数据放置到设备上
        images, captions, clip_image = batch[0], batch[1], batch[-1]
        clip_image = clip_image.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        images = images.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        original_images = torch.clone(images)

        # 前向传播和重建过程
        with torch.cuda.amp.autocast(dtype=torch.float16):  
            reconstructed_images, _, token_mask = accelerator.unwrap_model(model)(images, clip_image=clip_image)[:3]
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        # 对生成图像和原始图像进行归一化（调整到 [0, 1]）

        rmse = F.mse_loss(reconstructed_images, original_images, reduction='none').mean(dim=[1, 2, 3])
        batch_hit_count = (rmse <= 0.02).int().sum()
        total_hit += batch_hit_count
        total_rMSE += rmse.sum().item()
        num_samples += images.size(0)  # 更新样本数量


        reconstructed_images = torch.clamp(255 * reconstructed_images, 0, 255).to(torch.uint8)
        original_images = torch.clamp(255 * original_images, 0, 255).to(torch.uint8)

        # FID 更新：torchmetrics 会自动处理特征提取和计算
        fid_metric.update(original_images, real=True)  # 更新真实图像
        fid_metric.update(reconstructed_images.squeeze(2), real=False)  # 更新生成图像

        total_tokens += token_mask.sum().item()
        
    # 模型恢复训练模式
    model.train()


    avg_rMSE = total_rMSE / num_samples if num_samples > 0 else 0.0
    rQPR = total_hit / num_samples if num_samples > 0 else 0.0

    # 返回最终的 FID 结果
    return fid_metric.compute().item(), total_tokens, avg_rMSE, rQPR



def test(config, model):
    # Setup loggers
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers=[kwargs],
    )

    accelerator.wait_for_everyone()

    """Creates data loader for testing."""
    val_dataset = instantiate_from_config(config.dataset.val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)


    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    
    pretrained_tokenizer = create_pretrained_tokenizer(config,
                                                       accelerator)
    
    eval_scores, total_tokens, avg_rMSE, rQPR = eval_reconstruction(model, val_dataloader, accelerator, pretrained_tokenizer)
    # import pdb; pdb.set_trace()
    if accelerator.is_main_process:
        print(f"rFID is {eval_scores:04f}, total tokens is {256 - total_tokens / len(val_dataset) * accelerator.num_processes}")
        print(f"avg RMSE is {avg_rMSE:04f}")
        print(f"rQPR is: {rQPR:04f}")




def main(args):
    config_path = args.config
    config = OmegaConf.load(config_path)
    config['config_path'] = config_path
    model = AdapTok(config)
    # checkpoint_path = config.evaluate.checkpoint_path
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(torch.load("/mnt/hwfile/ai4earth/luxiaocheng/project/Generation/EfficientGen/FlexGen/adaptok_s_256_stage1_randmask_500k/checkpoint-500000/ema_model/pytorch_model.bin"), strict=True)
    test(config, model)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing Config')
    parser.add_argument('--config', default='configs/training/tokenizer/titok/imagenet.yaml', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)