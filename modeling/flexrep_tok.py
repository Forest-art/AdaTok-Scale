"""This file contains the model definition of TiTok.

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
import torch.nn as nn
from einops import rearrange

import timm
from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
from omegaconf import OmegaConf
from pathlib import Path
import cv2
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F
import open_clip

class SimplePolicyNet(nn.Module):
    def __init__(self, 
                 input_dim=512, 
                 hidden_dim=512,
                 num_positions=256,
                 temperature=1.0):
        super().__init__()
        
        # 核心映射网络
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_positions)
        )
        
        # 可学习的位置偏置
        self.pos_bias = nn.Parameter(torch.zeros(num_positions))
        
        # 自适应温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        """输入: [B, 512], 输出: [B, 256]"""
        # 基础映射
        logits = self.mapping(x)  # [B, 256]
        
        # 添加位置偏置
        logits = logits + self.pos_bias  # 广播机制自动扩展
        
        # 温度调节
        scaled_logits = logits / self.temperature.clamp(min=0.1, max=10.0)
        
        return scaled_logits

class CLIPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = SimplePolicyNet()
        
    def forward(self, clip_features):
        logits = self.policy_net(clip_features)
        dist = torch.distributions.Categorical(logits=logits)

        # 采样长度 (1-256)
        length = dist.sample() + 1
        return length, dist.log_prob(length-1), dist.entropy()


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class FlexRep(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)
        self.use_policynet = config.model.vq_model.get("use_policynet", False)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))

        # self.fud_model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
        # self.linear_proj = torch.nn.Conv2d(1024, 12, kernel_size=1, bias=True)

        if self.use_policynet:    
            self.clip_model = open_clip.create_model_and_transforms(model_name="ViT-B-16-quickgelu", pretrained="metaclip_400m")[0]

            for p in self.parameters():
                p.requires_grad = False

            self.policy_net = CLIPPolicy()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def get_key_padding_mask(self, mask_length_start=257, mask_length_end=513, batch_size=1, mask_lengths=None):
        if mask_lengths is None:
            mask_lengths = torch.randint(mask_length_start+1, mask_length_end, (batch_size,))
        key_padding_mask = torch.zeros(batch_size, mask_length_end, dtype=torch.bool)
        seq_length = key_padding_mask.size(1)  # 获取序列长度
        indices = torch.arange(seq_length).to(mask_lengths.device)  # 创建一个从 0 到 seq_length-1 的索引张量
        key_padding_mask = indices >= mask_lengths.unsqueeze(1)  # 广播比较，生成 mask
        return key_padding_mask
    

    def encode(self, x, token_length=None, clip_image=None):
        import random
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)

                # attn_mask, token_mask = self.gen_attn_mask(x)
                if self.use_policynet:
                    image_features = self.clip_model.encode_image(clip_image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    keep_lengths, log_prob, entropy_loss = self.policy_net(image_features)

                else:
                    keep_lengths = torch.randint(1, self.num_latent_tokens, (z.shape[0],))
                if token_length is not None:
                    keep_lengths = torch.clamp(keep_lengths, token_length, token_length)
                key_padding_mask = self.get_key_padding_mask(
                    257, 257 + self.num_latent_tokens, z.shape[0], mask_lengths=257 + keep_lengths
                )

                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            # attn_mask, token_mask = self.gen_attn_mask(x)
            keep_lengths = torch.randint(0, 256, (z.shape[0],))
            if token_length is not None:
                keep_lengths = torch.clamp(keep_lengths, token_length, token_length)
            key_padding_mask = self.get_key_padding_mask(
                257, 257 + self.num_latent_tokens, z.shape[0], mask_lengths=257 + keep_lengths
            )
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        if self.use_policynet:
            return z_quantized, result_dict, key_padding_mask.to(z.device), log_prob, entropy_loss.mean(), keep_lengths
        else:
            return z_quantized, result_dict, key_padding_mask.to(z.device)
    
    def forward_dinov2(self, x):
        x = 2 * x - 1
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.fud_model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        x = self.linear_proj(x)
        return x
    
    def decode(self, z_quantized, key_padding_mask):
        decoded = self.decoder(z_quantized, key_padding_mask)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x, token_length=None, clip_image=None):
        if self.use_policynet:
            z_quantized, result_dict, key_padding_mask, log_prob, entropy_loss, keep_lengths = self.encode(x, token_length, clip_image)
        else:
            z_quantized, result_dict, key_padding_mask = self.encode(x, token_length=token_length)
        decoded = self.decode(z_quantized, key_padding_mask)

        if self.use_policynet:
            return decoded, result_dict, key_padding_mask, log_prob, entropy_loss, keep_lengths
       
        result_dict['z_quantized'] = z_quantized

        return decoded, result_dict, key_padding_mask