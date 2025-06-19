import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

from omegaconf import OmegaConf
import os
import time
import argparse
from modeling.titok import TiTok
from modeling.language.t5 import T5Embedder
from modeling.generator.transformer import GPT_models
from modeling.generator.generate import generate
# from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("configs/training/TiTok/stage3/laion_s128.yaml")
    # create and load model
    tokenizer = TiTok(config)
    tokenizer.to(device)
    tokenizer.eval()
    checkpoint = torch.load("/mnt/petrelfs/luxiaocheng/project/Generation/EfficientGen/FlexGen/exp/titok_s_128_stage2_laion_run1/checkpoint-40000/ema_model/pytorch_model.bin", map_location="cpu")
    tokenizer.load_state_dict(checkpoint, strict=True)
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=4096,
        block_size=16 ** 2,
        num_classes=1000,
        cls_token_num=120,
        model_type="t2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        token_dropout_p=0.1,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load("/mnt/petrelfs/luxiaocheng/project/Generation/EfficientGen/FlexGen/exp/titok_s_128_stage3_laion_run1/checkpoint-100000/unwrapped_model/pytorch_model.bin", map_location="cpu")

    gpt_model.load_state_dict(checkpoint, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")
        
    t5_model = T5Embedder(
        device=device, 
        local_cache=False, 
        cache_dir='pretrained_models/t5-ckpt', 
        dir_or_name='flan-t5-xl',
        torch_dtype=torch.float16,
        model_max_length=120,
    )
    prompts = [
        "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
        "A blue Porsche 356 parked in front of a yellow brick wall.",
        "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
        "The American flag and a city skyline"
    ]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, 128, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    t2 = time.time()
    samples = tokenizer.decode_tokens(index_sample)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    save_image(samples, "./demos/sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(0, 1))
    print(f"image is saved to demos/sample_{args.gpt_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--codebook-size", type=int, default=4096, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=12, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
