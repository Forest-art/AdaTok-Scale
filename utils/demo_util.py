"""Demo file for sampling images from TiTok.

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


import torch

from omegaconf import OmegaConf
from modeling.adaptmask import AdapTok
from modeling.titok import TiTok
from modeling.tatitok import TATiTok
from modeling.maskgit import ImageBert, UViTBert, EntropyUViTBert
from modeling.rar import RAR


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def get_adaptok_tokenizer(config):
    tokenizer = AdapTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"), strict=True)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_tokenizer(config):
    tokenizer = TiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"), strict=True)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_tatitok_tokenizer(config):
    tokenizer = TATiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"), strict=True)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_generator(config):
    if config.model.generator.model_type == "ViT":
        model_cls = ImageBert
    elif config.model.generator.model_type == "UViT":
        model_cls = UViTBert
    elif config.model.generator.model_type == "EntropyUViT":
        model_cls = EntropyUViTBert
    else:
        raise ValueError(f"Unsupported model type {config.model.generator.model_type}")
    generator = model_cls(config)
    weight = torch.load(config.experiment.generator_checkpoint, map_location="cpu")
    # keys_to_remove = ["embeddings.weight", "pos_embed"]  # 需要忽略的 key
    # for key in keys_to_remove:
    #     if key in weight:
    #         del weight[key]
    generator.load_state_dict(weight, strict=True)
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_rar_generator(config):
    model_cls = RAR
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.set_random_ratio(0)
    return generator


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    # if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
    # labels = [292, 285, 275, 751, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image


from modeling.generator.generate import generate

@torch.no_grad()
def sample_fn_llamagen(generator,
              tokenizer,
              labels=None,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    # if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
    # labels = [292, 285, 275, 751, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generate(generator, labels, 256, cfg_scale=1.5, cfg_interval=-1, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True)
    
    key_padding_mask = generated_tokens == 4096
    keep_lengths = 256 - key_padding_mask.sum(dim=1)
    key_padding_mask = tokenizer.get_key_padding_mask(257, 513, keep_lengths.shape[0], mask_lengths=257 + keep_lengths)

    generated_tokens[generated_tokens==4096] = 0

    generated_image = tokenizer.decode_tokens(generated_tokens.view(generated_tokens.shape[0], -1), key_padding_mask)
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image
