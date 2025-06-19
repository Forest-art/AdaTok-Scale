import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import math
import random
import torch
import numpy as np
from PIL import Image
import open_clip
from datasets import load_dataset
from transformers import CLIPImageProcessor

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


# class HuggingfaceImageNet(Dataset):
#     def __init__(self, root, mode, vf_type=None, image_size=256):
#         """
#         Custom Dataset for ImageNet-like structure.

#         Args:
#             root (str): Path to the root directory of the dataset.
#             mode (str): One of 'train', 'val', or 'test'.
#             image_size (int): Size to which images will be resized.
#             transform (torchvision.transforms.Compose): Optional transform to apply to images.
#         """
#         assert mode in ['train', 'validation', 'test'], "Mode must be 'train', 'validation', or 'test'."
#         self.dataset = load_dataset(root)[mode]

#         self.vf_type = vf_type

#         if mode == "train":
#             self.transform = transforms.Compose([
#                 transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
#             ])
        
#         self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name="ViT-B-16-quickgelu", pretrained="metaclip_400m")

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         image_pair = self.dataset[index]

#         # Get the original image and convert to RGB
#         original_image = image_pair['image'].convert("RGB")

#         clip_image = self.preprocess(original_image)
#         # Custom transform for training or evaluation
#         image = self.transform(original_image)
#         label = image_pair['label']

#         return image, label, clip_image


class HuggingfaceImageNet(Dataset):
    def __init__(self, root, mode, vf_type=None, image_size=256):
        """
        Custom Dataset for ImageNet-like structure.

        Args:
            root (str): Path to the root directory of the dataset.
            mode (str): One of 'train', 'val', or 'test'.
            image_size (int): Size to which images will be resized.
            transform (torchvision.transforms.Compose): Optional transform to apply to images.
        """
        assert mode in ['train', 'validation', 'test'], "Mode must be 'train', 'validation', or 'test'."
        self.dataset = load_dataset(root)[mode]

        self.vf_type = vf_type

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
            ])
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name="ViT-B-16-quickgelu", pretrained="metaclip_400m")

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, index):
        image_pair = self.dataset[index]

        # Get the original image and convert to RGB
        original_image = image_pair['image'].convert("RGB")

        clip_image = self.preprocess(original_image)
        # Custom transform for training or evaluation
        image = self.transform(original_image)
        label = image_pair['label']

        return image, label, clip_image
    


class ImageNetDataset(Dataset):
    def __init__(self, root, mode, image_size=256, transform=None):
        """
        Custom Dataset for ImageNet-like structure.

        Args:
            root (str): Path to the root directory of the dataset.
            mode (str): One of 'train', 'val', or 'test'.
            image_size (int): Size to which images will be resized.
            transform (torchvision.transforms.Compose): Optional transform to apply to images.
        """
        import json
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.root = root
        self.mode = mode
        self.image_size = image_size
        with open("./data/class_to_index.json", "r") as f:
            self.class_index = json.load(f)
        # If no transform is provided, define a default transform
        if self.mode == "train":
            self.transform = transform or transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1], inplace=True)
            ])

        # Load image paths
        self.image_paths, self.labels = self._load_image_paths()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name="ViT-B-16-quickgelu", pretrained="metaclip_400m")

    def _load_image_paths(self):
        """
        Load all image paths based on the dataset structure.
        - For 'train': Recursively load all images from subdirectories.
        - For 'val' and 'test': Load all images directly from the root directory.

        Returns:
            list: List of image file paths.
        """
        image_paths, labels = [], []
        if self.mode == 'train':
            with open(os.path.join(self.root, "meta", "train.txt"), 'r') as f:
                datas = f.readlines()
            for data in datas:
                data = data.strip()
                image_path = os.path.join(self.root, "train", data.split(" ")[0])
                label = data.split(" ")[1]
                image_paths.append(image_path)
                labels.append(int(label))

        else:
            with open(os.path.join(self.root, "meta", "val.txt"), 'r') as f:
                datas = f.readlines()
            for data in datas:
                data = data.strip()
                image_path = os.path.join(self.root, "val", data.split(" ")[0])
                label = data.split(" ")[1]
                image_paths.append(image_path)
                labels.append(int(label))

        return image_paths, labels
    
    def __len__(self):
        if self.mode == "train":
            return len(self.image_paths)
        else:
            return len(self.image_paths)
        
    def get_infordensity(self, x, patch_size=16):
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
        return np.clip(density_map.mean(), 0, 200).astype(np.uint8) // 20, np.clip(density_map.mean(), 0, 200)


    def __getitem__(self, idx):
        """
        Load an image and apply transforms.

        Args:
            idx (int): Index of the image to load.

        Returns:
            Tensor: Transformed image tensor.
        """
        while True:  # Keep trying until a valid sample is found
            try:
                img_path = self.image_paths[idx]
                label = self.labels[idx]
                
                # Load and convert image to RGB
                image = Image.open(img_path).convert("RGB")

                clip_image = self.preprocess(image)
                
                # Apply transformations if any
                if self.transform:
                    image = self.transform(image)
                
                # Calculate entropy level and entropy
                entropy_level, entropy = self.get_infordensity(image.permute(1, 2, 0), patch_size=256)
                
                return [image, label, entropy_level, entropy, img_path, clip_image]
            
            except Exception as e:
                print(f"Error processing image at index {idx} ({img_path}): {e}")
                # Skip this sample and try the next one
                idx = (idx + 1) % len(self.image_paths)  # Move to the next index, wrap around if necessary