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
from modeling.flexrep_tok import FlexRep
from utils.base_utils import instantiate_from_config
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from data.imagenet import ImageNetDataset, HuggingfaceImageNet
import time
import torchvision.utils as vutils

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
    total_reward = 0
    total_tokens = 0
    model.eval()
    # 初始化 FID 指标（torchmetrics）
    fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
    
    # 确定是否为主进程
    is_main_process = accelerator.is_main_process

    # 初始化进度条，仅在主进程显示
    progress_bar = tqdm(eval_loader, desc="Evaluating Reconstruction", leave=True, disable=not is_main_process)

    for batch_idx, batch in enumerate(progress_bar):
        # 将批量数据放置到设备上
        images, captions = batch[0], batch[1]
        images = images.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        original_images = torch.clone(images)
        clip_images = batch[-1]

        all_reconstructed = []  # 用于存储所有 keep_length 的重建图像

        for keep_length in [2, 4, 8, 16, 32, 64, 128, 256]:

            # 前向传播和重建过程
            with torch.cuda.amp.autocast(dtype=torch.float16):  
                reconstructed_images, _, token_mask = accelerator.unwrap_model(model)(images, token_length=keep_length, clip_image=clip_images)[:3]
            if pretrained_tokenizer is not None:
                reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))

            # 对生成图像和原始图像进行归一化（调整到 [0, 1]）
            rmse = F.mse_loss(reconstructed_images, original_images, reduction='none').mean(dim=[1, 2, 3])
            batch_hit_count = (rmse <= 0.02).int().sum()
            total_hit += batch_hit_count
            total_rMSE += rmse.sum().item()
            num_samples += images.size(0)  # 更新样本数量

            is_satisfied = (rmse < 0.02)
            tl = 256 - token_mask.sum(dim=1)
            rl = rmse

            # 奖励计算分三个层次
            reward = torch.where(
                is_satisfied,
                # 层次1：满足条件的标准奖励
                1 + (256 - tl) / 255,  # 归一化[0,1]
                - (rl - 0.02) * 10 + tl/255  # 综合惩罚
            )
            total_reward += reward.sum().item()
            reconstructed_images = torch.clamp(255 * reconstructed_images, 0, 255).to(torch.uint8)
            original_images1 = torch.clamp(255 * original_images, 0, 255).to(torch.uint8)

            all_reconstructed.append(reconstructed_images)

            # FID 更新：torchmetrics 会自动处理特征提取和计算
            fid_metric.update(original_images1, real=True)  # 更新真实图像
            fid_metric.update(reconstructed_images.squeeze(2), real=False)  # 更新生成图像

            total_tokens += token_mask.sum().item()
        

        all_reconstructed = torch.cat(all_reconstructed, dim=3)
        combined = torch.cat((all_reconstructed, original_images1), dim=3)

        import pdb; pdb.set_trace()

        save_path = f"samples/batch_{batch_idx}.png"
        vutils.save_image(combined[0].float() / 255.0, save_path)
        

    # 模型恢复训练模式
    model.train()

    avg_rMSE = total_rMSE / num_samples if num_samples > 0 else 0.0
    rQPR = total_hit / num_samples if num_samples > 0 else 0.0
    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0

    # 返回最终的 FID 结果
    return fid_metric.compute().item(), total_tokens, avg_rMSE, rQPR, avg_reward



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
    val_dataset = HuggingfaceImageNet("ILSVRC/imagenet-1k", "test")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    
    pretrained_tokenizer = create_pretrained_tokenizer(config,
                                                       accelerator)
    
    eval_scores, total_tokens, avg_rMSE, rQPR, avg_reward = eval_reconstruction(model, val_dataloader, accelerator, pretrained_tokenizer)
    # import pdb; pdb.set_trace()
    if accelerator.is_main_process:
        print(f"rFID is {eval_scores:04f}, total tokens is {256 - total_tokens / len(val_dataset) * accelerator.num_processes}")
        print(f"avg RMSE is {avg_rMSE:04f}")
        print(f"rQPR is: {rQPR:04f}")
        print(f"avg_reward is: {avg_reward:04f}")


def main(args):
    config_path = args.config
    config = OmegaConf.load(config_path)
    config['config_path'] = config_path
    model = FlexRep(config)

    # checkpoint_path = config.evaluate.checkpoint_path
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(torch.load("./pretrained_models/flexrep_s256_stage2_200k_randmask.bin"), strict=True)

    test(config, model)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing Config')
    parser.add_argument('--config', default='configs/training/FlexRep/stage2/imagenet_s256.yaml', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)