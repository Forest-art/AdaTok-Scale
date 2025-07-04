"""Training script for MaskGIT with TiTok.

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
"""
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger
from torch.utils.data import Dataset, DataLoader
from utils.train_utils import (
    get_config, create_model_and_loss_module, get_titok_tokenizer,
    create_optimizer, create_lr_scheduler, create_dataloader,
    auto_resume, save_checkpoint, 
    train_one_epoch_generator)
from utils.base_utils import instantiate_from_config

def main():
    workspace = os.environ.get('WORKSPACE', '')
    torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
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

    logger = setup_logger(name="TiTok-Gen", log_level="INFO",
     output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    tokenizer = get_titok_tokenizer(config)
    tokenizer.to(accelerator.device)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="maskgit")

    # model.load_state_dict(torch.load("./pretrained_models/generator_titok_s128.bin", map_location='cpu'), strict=True)

    # model = torch.compile(model)

    # load state dict from the model
    weight = torch.load("./pretrained_models/generator_titok_s128.bin", map_location='cpu')
    keys_to_remove = ["embeddings.weight", "pos_embed"]  # 需要忽略的 key
    for key in keys_to_remove:
        if key in weight:
            del weight[key]
    model.load_state_dict(weight, strict=False)
    weight_init = torch.load("./pretrained_models/generator_titok_s128.bin", map_location='cpu')
    model.embeddings.weight.data[:5098] = weight_init["embeddings.weight"]
    model.pos_embed[:, :1, :].data = weight_init["pos_embed"][:, :1, :]
    model.pos_embed[:, 2:, :].data = weight_init["pos_embed"][:, 1:, :]


    # model.load_state_dict(torch.load("./pretrained_models/generator_titok_s128.bin", map_location='cpu'), strict=True)

    optimizer, _ = create_optimizer(config, logger, model, loss_module,
                                    need_discrminator=False)

    lr_scheduler, _ = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer=None)

    train_dataset = instantiate_from_config(config.dataset.train)
    val_dataset = instantiate_from_config(config.dataset.val)
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.per_gpu_batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.per_gpu_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
        # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch_generator(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer,
                            lr_scheduler,
                            train_dataloader,
                            tokenizer,
                            global_step,
                            model_type="maskgit")
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()