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
from modeling.flextitok import FlexTiTok
import warnings
from utils.base_utils import instantiate_from_config
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn.functional as F
import numpy as np

def get_infordensity(x, patch_size=16):
    import cv2
    x = torch.clamp(255 * x, 0, 255).to(torch.uint8).cpu().numpy()
    gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if len(x.shape) == 3 else x

    # 获取图像的梯度（使用 Sobel 算子）
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # x 方向梯度
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # y 方向梯度
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)  # 梯度幅值

    # 图像尺寸
    h, w = gray_image.shape

    # 计算每个 patch 的信息密度
    density_map = []
    for i in range(0, h, patch_size):
        row_density = []
        for j in range(0, w, patch_size):
            # 提取 patch 的梯度幅值
            patch = gradient_magnitude[i:i+patch_size, j:j+patch_size]
            # 计算信息密度（梯度幅值的均值）
            density = patch.mean()
            row_density.append(density)
        density_map.append(row_density)
    
    # 转为 numpy 数组
    density_map = np.array(density_map)
    return np.clip(density_map.mean(), 0, 200)


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    pretrained_tokenizer=None
):
    # 评估模式
    model.eval()
    # 初始化 FID 指标（torchmetrics）
    fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
    
    # 初始化 PSNR 变量
    total_psnr = 0.0
    num_samples = 0
    total_rMSE = 0.0

    # 确定是否为主进程
    is_main_process = accelerator.is_main_process

    # 初始化进度条，仅在主进程显示
    progress_bar = tqdm(eval_loader, desc="Evaluating Reconstruction", leave=True, disable=not is_main_process)

    for batch in progress_bar:
        # 将批量数据放置到设备上
        images, captions, entropy = batch[0], batch[1], batch[3]
        images = images.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        original_images = torch.clone(images)

        # 前向传播和重建过程
        with torch.cuda.amp.autocast(dtype=torch.float16):  
            reconstructed_images, _ = accelerator.unwrap_model(model)(images)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))

        # 对生成图像和原始图像进行归一化（调整到 [0, 1]）
        reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
        original_images = torch.clamp(original_images, 0, 1)

        # 计算 PSNR
        mse = F.mse_loss(reconstructed_images, original_images)  # 计算均方误差
        psnr = -10.0 * torch.log10(mse)  # 计算 PSNR
        total_psnr += psnr.item() * images.size(0)  # 累加 PSNR
        num_samples += images.size(0)  # 更新样本数量

        rmse = F.mse_loss(reconstructed_images, original_images)
        total_rMSE += rmse.item() * images.size(0)

        # FID 更新：torchmetrics 会自动处理特征提取和计算
        reconstructed_images = torch.clamp(255 * reconstructed_images, 0, 255).to(torch.uint8)
        original_images = torch.clamp(255 * original_images, 0, 255).to(torch.uint8)

        fid_metric.update(original_images, real=True)  # 更新真实图像
        fid_metric.update(reconstructed_images.squeeze(2), real=False)  # 更新生成图像

    # 模型恢复训练模式
    model.train()

    # 收集所有 GPU 的 PSNR 和样本数量
    total_rMSE = torch.tensor(total_rMSE, device=accelerator.device)
    total_psnr = torch.tensor(total_psnr, device=accelerator.device)
    num_samples = torch.tensor(num_samples, device=accelerator.device)
    total_psnr = accelerator.gather(total_psnr).sum().item()  # 收集并求和
    num_samples = accelerator.gather(num_samples).sum().item()  # 收集并求和

    # 计算全局平均 PSNR
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0

    # 计算平均信息熵MSE
    avg_rMSE = total_rMSE / num_samples if num_samples > 0 else 0.0

    # 收集 FID 结果
    fid_score = fid_metric.compute().item()

    # 返回最终的 FID 结果和 PSNR
    return fid_score, avg_psnr, avg_rMSE


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
    
    eval_scores, avg_psnr, avg_rMSE = eval_reconstruction(model, val_dataloader, accelerator, pretrained_tokenizer)
    # import pdb; pdb.set_trace()
    if accelerator.is_main_process:
        print(f"rFID is {eval_scores:04f}")
        print(f"Average PSNR is {avg_psnr:04f}")
        print(f"Average rMSE is {avg_rMSE:04f}")



def main(args):
    config_path = args.config
    config = OmegaConf.load(config_path)
    config['config_path'] = config_path
    model = TiTok(config)
    # checkpoint_path = config.evaluate.checkpoint_path
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(torch.load("./pretrained_models/tokenizer_titok_s128.bin"), strict=True)
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