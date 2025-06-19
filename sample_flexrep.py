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
from PIL import Image


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
        images, captions = batch[0], batch[1]
        images = images.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        original_images = torch.clone(images)

        # 前向传播和重建过程
        with torch.cuda.amp.autocast(dtype=torch.float16):  
            reconstructed_images, _, token_mask = accelerator.unwrap_model(model)(images, token_length=256)
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

    images = Image.open("assets/demo1.png").convert("RGB")

    from torchvision import transforms
    from data.imagenet import center_crop_arr
    transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
            ])
    images = transform(images).cuda()
    model = model.cuda()
    reconstructed_images, _, token_mask = model(images.unsqueeze(0), token_length=256)
    rmse = F.mse_loss(reconstructed_images, images, reduction='none').mean(dim=[1, 2, 3])
    print(rmse)




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