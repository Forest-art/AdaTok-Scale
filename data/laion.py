import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import math
import random
import torch
import numpy as np
from data.s3_client import s3_client
from data.imagenet import center_crop_arr, random_crop_arr

class LaionCOCODataset(Dataset):
    def __init__(self, s3_cfg, mode, image_size=256):
        self.s3 = s3_client(**s3_cfg)
        self.train_df = None
        self.mode = mode
        self.image_size = image_size

        # If no transform is provided, define a default transform
        if self.mode == "train":
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
        self._load_image_paths()

        if self.train_df is not None:
            print("Combined DataFrame:")
            print(self.train_df.head())
            print(f"Total rows: {len(self.train_df)}")

    def _load_image_paths(self):
        """
        Load all image paths based on the dataset structure.
        - For 'train': Recursively load all images from subdirectories.
        - For 'val' and 'test': Load all images directly from the root directory.

        Returns:
            list: List of image file paths.
        """
        if self.mode == 'train':
            # Recursively find all images in subdirectories
            self.train_df = self.s3.read_all_parquet_files(max_workers=16, clips=[0, 5])
            self.image_paths = self.train_df
        else:
            # Directly find all images in the root directory
            self.eval_df = self.s3.read_all_parquet_files(max_workers=16, clips=[10, 11])
            self.image_paths = self.eval_df

    def __len__(self):
        if self.mode=="train":
            return len(self.image_paths)
        else:
            return len(self.image_paths) // 50

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
        img_path = self.image_paths.iloc[idx]["URL"]
        img_cap = self.image_paths.iloc[idx]["top_caption"]

        try:
            # Attempt to read the image
            image = self.s3.read_image_from_s3(img_path).convert("RGB")  # Ensure RGB format
            if self.transform:
                image = self.transform(image)
            entropy_level, entropy = self.get_infordensity(image.permute(1, 2, 0), patch_size=256)
            return [image, img_cap, entropy_level, entropy, img_path]
        except:
            # Log the error and skip this image
            # Return None or skip this item by raising IndexError
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Move to the next index




class Text2ImageLaion(Dataset):
    def __init__(self, s3_cfg, mode, image_size=256):
        self.s3 = s3_client(**s3_cfg)
        self.train_df = None
        self.mode = mode
        self.image_size = image_size

        # If no transform is provided, define a default transform
        if self.mode == "train":
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
        self._load_image_paths()

        if self.train_df is not None:
            print("Combined DataFrame:")
            print(self.train_df.head())
            print(f"Total rows: {len(self.train_df)}")


    def _load_image_paths(self):
        """
        Load all image paths based on the dataset structure.
        - For 'train': Recursively load all images from subdirectories.
        - For 'val' and 'test': Load all images directly from the root directory.

        Returns:
            list: List of image file paths.
        """
        if self.mode == 'train':
            # Recursively find all images in subdirectories
            self.train_df = self.s3.read_all_parquet_files(max_workers=16, clips=[0, 10])
            self.image_paths = self.train_df
        else:
            # Directly find all images in the root directory
            self.eval_df = self.s3.read_all_parquet_files(max_workers=16, clips=[10, 11])
            self.image_paths = self.eval_df

    def __len__(self):
        if self.mode=="train":
            return len(self.image_paths)
        else:
            return len(self.image_paths) // 50


    def __getitem__(self, idx):
        img_path = self.image_paths.iloc[idx]["URL"]
        img_cap = self.image_paths.iloc[idx]["top_caption"]

        try:
            # Attempt to read the image
            image = self.s3.read_image_from_s3(img_path).convert("RGB")  # Ensure RGB format
            if self.transform:
                image = self.transform(image)

            return [image, img_cap]
        except (UnidentifiedImageError, KeyError, AttributeError, IOError) as e:
            # Log the error and skip this image
            # Return None or skip this item by raising IndexError
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Move to the next index



if __name__ == "__main__":
    s3_cfg=dict(access_key='FB7QKWTWP279SQMLBX4H',
        secret_key= 'dN6ph2f9cQcVhnOCngiGKwPUjMqpM9o4oiKM67mb',
        bucket_name='public-dataset',
        data_prefix = "laion-coco/meta/",
        endpoint='http://p-ceph-norm-outside.pjlab.org.cn')

    train_dataset = LaionCOCODataset(s3_cfg, "train", 256)