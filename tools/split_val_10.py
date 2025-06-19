import os
from PIL import Image
import numpy
import numpy as np
import torch

def get_infordensity(x, patch_size=16):
    import cv2
    # x = torch.clamp(255 * x, 0, 255).to(torch.uint8).cpu().numpy()
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
    return np.clip(density_map.mean(), 0, 200).astype(np.uint8) // 20

# 主函数
def classify_images_by_density(image_dir, output_dir, patch_size=16):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建 11 个子文件夹
    for i in range(11):
        subfolder = os.path.join(output_dir, f'density_{i}')
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    import cv2
    # 遍历图片目录
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(image_dir, filename)
            try:
                image = cv2.imread(image_path)  # 替换为你的图像路径

                resized_image = cv2.resize(image, (256, 256))
                # 打开图片=
                # 计算信息密度等级
                density_level = get_infordensity(resized_image, patch_size)

                # 目标子文件夹
                target_folder = os.path.join(output_dir, f'density_{density_level}')

                # 保存图片到目标文件夹
                target_path = os.path.join(target_folder, filename)
                cv2.imwrite(target_path, resized_image)
                print(f'Moved {filename} to {target_folder}')
            except Exception as e:
                print(f'Error processing {filename}: {e}')

# 使用示例
if __name__ == '__main__':
    image_dir = '/mnt/hwfile/ai4earth/luxiaocheng/dataset/imagenet1k/val'  # 替换为你的图片目录
    output_dir = '/mnt/hwfile/ai4earth/luxiaocheng/dataset/imagenet1k/val_density'    # 替换为输出目录
    classify_images_by_density(image_dir, output_dir)