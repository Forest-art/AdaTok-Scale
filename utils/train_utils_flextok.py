"""Training utils for TiTok.

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
import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
# import open_clip

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2, ReconstructionLoss_Single_Stage, MLMLoss, ARLoss, AdaptiveLoss_Stage1, AdaptiveLoss_Stage2
from modeling.titok import TiTok, PretrainedTokenizer
from modeling.tatitok import TATiTok
from modeling.adaptmask import AdapTok
from modeling.one_d_piece import OneDPiece
from modeling.flextitok import FlexTiTok
from modeling.flextok_entropy import EntropyFlexTiTok
# from modeling.maskgit import ImageBert, UViTBert, EntropyUViTBert
# from modeling.rar import RAR, EntropyRAR
from evaluator import VQGANEvaluator
from utils.demo_util import get_titok_tokenizer, sample_fn
from torchmetrics.image.fid import FrechetInceptionDistance
from imagenet_classes import imagenet_idx2classname
from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation
from modeling.flexrep_tok import FlexRep


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_tokenizer(config, accelerator=None):
    if config.model.vq_model.finetune_decoder:
        # No need of pretrained tokenizer at stage2
        pretrianed_tokenizer = None
    else:
        pretrianed_tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
        if accelerator is not None:
            pretrianed_tokenizer.to(accelerator.device)
    return pretrianed_tokenizer



def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok"):
    """Creates TiTok model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "flexrep-tok":
        model_cls = FlexRep
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1  
    elif model_type == "one-d-piece":
        model_cls = OneDPiece
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1    
    elif model_type == "flextitok":
        model_cls = FlexTiTok
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1      
    elif model_type == "entropyflextitok":
        model_cls = EntropyFlexTiTok
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1        
    elif model_type == "adaptok":
        model_cls = AdapTok
        loss_cls = AdaptiveLoss_Stage2 
    elif model_type == "titok":
        model_cls = TiTok
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1 
    elif model_type == "tatitok":
        model_cls = TATiTok
        loss_cls = ReconstructionLoss_Single_Stage
    # elif model_type == "maskgit":
    #     if config.model.generator.model_type == "ViT":
    #         model_cls = ImageBert
    #     elif config.model.generator.model_type == "UViT":
    #         model_cls = UViTBert
    #     elif config.model.generator.model_type == "EntropyUViT":
    #         model_cls = EntropyUViTBert
    #     else:
    #         raise ValueError(f"Unsupported generator model_type {config.model.generator.model_type}")
        # loss_cls = MLMLoss
    # elif model_type == "rar":
    #     model_cls = RAR
    #     loss_cls = ARLoss
    # else:
    #     raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        if config.model.vq_model.finetune_decoder:
            # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
            pretrained_tokenizer_weight = torch.load(
                config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
            )
            # Only keep the quantize and decoder part
            pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
            model_weight.update(pretrained_tokenizer_weight)
        
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config)

    # Print Model for sanity check.
    # if accelerator.is_main_process:
    #     if model_type in ["titok", "flextitok", "one-d-piece", "entropyflextitok"]:
    #         input_size = (1, 3, config.dataset.train.params.image_size, config.dataset.train.params.image_size)
    #         model_summary_str = summary(model, input_size=input_size, depth=5,
    #         col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["tatitok"]:
    #         input_image_size  = (1, 3, config.dataset.train.params.image_size, config.dataset.train.params.image_size)
    #         input_text_size = (1, 77, 768)
    #         input_size = [input_image_size, input_text_size]
    #         model_summary_str = summary(model, input_size=input_size, depth=5,
    #         col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["maskgit", "rar"]:
    #         input_size = (1, config.model.vq_model.num_latent_tokens)
    #         input_data = [
    #             torch.randint(0, config.model.vq_model.codebook_size, input_size),
    #             torch.ones(1, dtype=int)
    #         ]
    #         model_summary_str = summary(
    #             model, input_data=input_data, depth=7,
    #             col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     else:
    #         raise NotImplementedError

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="titok", need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    if (config.model.vq_model.finetune_decoder or model_type == "tatitok") and need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for TiTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=True,
            num_codebook_entries=config.model.vq_model.codebook_size
        )
    elif config.model.vq_model.get("quantize_mode", "vq") == "vae":
        evaluator = VQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False,
        )
    else:
        raise NotImplementedError
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    global_step,
                    model_type="titok",
                    clip_tokenizer=None,
                    clip_encoder=None,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        # if "image" in batch:
        images = batch[0].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        labels = batch[1].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        if "text" in batch and model_type == "tatitok":
            text = batch["text"]
            with torch.no_grad():
                text_guidance = clip_tokenizer(text).to(accelerator.device)
                cast_dtype = clip_encoder.transformer.get_cast_dtype()
                text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
                text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
                text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
                text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
                text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
                text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

        fnames = batch[1]
        data_time_meter.update(time.time() - end)

        # Obtain proxy codes
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(images)
        else:
            proxy_codes = None

        dino_feat = model.module.forward_dinov2(images)

        with accelerator.accumulate([model, loss_module]):
            if model_type == "titok":
                reconstructed_images, extra_results_dict, key_padding_mask = model(images)
                if proxy_codes is None:
                    autoencoder_loss, loss_dict = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step,
                        mode="generator",
                    )
                else:
                    autoencoder_loss, loss_dict = loss_module(
                        proxy_codes,
                        reconstructed_images,
                        extra_results_dict,
                        dino_feat
                    )    
            elif model_type == "tatitok":
                reconstructed_images, extra_results_dict = model(images, text_guidance)
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )
            else:
                raise NotImplementedError

            if accelerator.is_main_process and (global_step + 1) % 50 == 0:
                print(256 - key_padding_mask.squeeze().sum(dim=-1))
            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if (config.model.vq_model.finetune_decoder or model_type == "tatitok") and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_images(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                    text_guidance=text_guidance[:config.training.num_generated_images] if model_type == "tatitok" else None,
                    pretrained_tokenizer=pretrained_tokenizer
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
                accelerator.wait_for_everyone()
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        model_type=model_type,
                        clip_tokenizer=clip_tokenizer,
                        clip_encoder=clip_encoder,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


def get_rar_random_ratio(config, cur_step):
    randomness_anneal_start = config.model.generator.randomness_anneal_start
    randomness_anneal_end = config.model.generator.randomness_anneal_end
    if cur_step < randomness_anneal_start:
        return 1.0
    elif cur_step > randomness_anneal_end:
        return 0.0
    else:
        return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start)


def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,
                    model_type="maskgit"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        # tokenize on the fly
        images = batch[0].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        conditions = batch[1].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        # Encode images on the flight.
        with torch.no_grad():
            tokenizer.eval()
            # if model_type in ["rar", "entropyrar"]:
            #     input_tokens = tokenizer.encode(images)
            # else:
            input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"].reshape(images.shape[0], -1)

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)


        if model_type == "maskgit":
            # Randomly masking out input tokens.
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
        elif model_type in ["rar", "entropyrar"]:
            unwrap_model.set_random_ratio(get_rar_random_ratio(config, global_step))
        else:
            raise NotImplementedError
            

        with accelerator.accumulate([model]):

            if model_type == "maskgit":

                entropy_condition = torch.clamp(batch[2], 0, 9) + 1000
                conditions=torch.cat([conditions.view(conditions.shape[0], -1), entropy_condition.view(conditions.shape[0], -1)], dim=1)

                logits = model(masked_tokens, conditions, cond_drop_prob=config.model.generator.class_label_dropout)
                loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            elif model_type == "rar":
                condition = unwrap_model.preprocess_condition(
                    conditions, cond_drop_prob=config.model.generator.class_label_dropout
                )
                logits, labels = model(input_tokens, condition, return_labels=True)
                loss, loss_dict = loss_module(logits, labels)
            elif model_type == "entropyrar":
                entropy_condition = torch.clamp(batch[2].to(conditions.device), 0, 9) + 1000
                conditions=torch.cat([conditions.view(conditions.shape[0], -1), entropy_condition.view(conditions.shape[0], -1)], dim=1)
                condition = unwrap_model.preprocess_condition(
                    conditions, cond_drop_prob=config.model.generator.class_label_dropout
                )
                logits, labels = model(input_tokens, condition, return_labels=True)
                loss, loss_dict = loss_module(logits, labels)
            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step

from tqdm import tqdm

@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    pretrained_tokenizer=None
):
    model.eval()
    fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
    # local_model = accelerator.unwrap_model(model)
    # 添加进度条
    progress_bar = tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:  # 使用 tqdm 包装 eval_loader
        images = batch[0].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        original_images = torch.clone(images)
        reconstructed_images, model_dict, _ = model(images)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # 对生成图像和原始图像进行归一化（调整到 [0, 1]）
        reconstructed_images = torch.clamp(255 * reconstructed_images, 0, 255).to(torch.uint8)
        original_images = torch.clamp(255 * original_images, 0, 255).to(torch.uint8)
        
        # FID 更新：torchmetrics 会自动处理特征提取和计算
        fid_metric.update(original_images, real=True)  # 更新真实图像
        fid_metric.update(reconstructed_images.squeeze(2), real=False)  # 更新生成图像

    eval_results = dict(
        rFID = fid_metric.compute().item()
    )

    model.train()
    return eval_results


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="titok", text_guidance=None, 
                    pretrained_tokenizer=None):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    
    reconstructed_images, _, _ = accelerator.unwrap_model(model)(original_images)
    if pretrained_tokenizer is not None:
        reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


# def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
#     save_path = Path(output_dir) / f"checkpoint-{global_step}"

#     state_dict = accelerator.get_state_dict(model)
#     if accelerator.is_main_process:
#         unwrapped_model = accelerator.unwrap_model(model)
#         unwrapped_model.save_pretrained_weight(
#             save_path / "unwrapped_model",
#             save_function=accelerator.save,
#             state_dict=state_dict,
#         )
#         json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
#         logger.info(f"Saved state to {save_path}")

#     accelerator.save_state(save_path)
#     return save_path


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # 保存模型状态
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    # 保存加速器状态
    accelerator.save_state(save_path)

    # 获取所有检查点并按创建时间排序
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=os.path.getmtime)

    # 如果检查点数量超过2个，删除最旧的检查点
    if len(checkpoints) > 2:
        for checkpoint in checkpoints[:-2]:  # 保留最近的两个检查点
            logger.info(f"Deleting old checkpoint: {checkpoint}")
            os.system(f"rm -rf {checkpoint}")  # 删除整个检查点目录

    return save_path

def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)