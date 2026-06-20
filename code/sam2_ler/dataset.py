"""
dataset.py — 钢材缺陷分割数据集加载器
支持 NEU-Seg / Severstal 两个开源数据集
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A


# 数据集元信息 (ASI / 训练 / 表格共用)
DATASET_META = {
    "neu_seg": {
        "defect_classes": ["patches", "inclusion", "scratches"],
        "class_id_to_name": {
            0: "background", 1: "patches", 2: "inclusion", 3: "scratches",
        },
        "defect_class_ids": [1, 2, 3],
    },
    "severstal": {
        "defect_classes": ["class1", "class2", "class3", "class4"],
        "class_id_to_name": {
            0: "background", 1: "class1", 2: "class2", 3: "class3", 4: "class4",
        },
        "defect_class_ids": [1, 2, 3, 4],
    },
}


def get_dataset_meta(name: str) -> dict:
    assert name in DATASET_META, f"未知数据集: {name}, 可选: {list(DATASET_META)}"
    return DATASET_META[name]


# ============================================================
#  NEU-Seg 数据集
#  3类缺陷: patches(1), inclusion(2), scratches(3)
#  图像: 200x200 灰度
# ============================================================
class NEUSegDataset(Dataset):
    """NEU-Seg 钢材表面缺陷分割数据集"""

    CLASS_NAMES = ["background", "patches", "inclusion", "scratches"]
    NUM_CLASSES = 4  # 含背景

    def __init__(self, root_dir, split="train", img_size=1024, train_ratio=0.8, seed=42):
        """
        Args:
            root_dir:    NEU-Seg 数据集根目录, 包含 images/ 和 annotations/
            split:       'train' 或 'val'
            img_size:    SAM2 输入尺寸 (默认 1024)
            train_ratio: 训练集比例 (仅在没有 training/test 子目录时使用)
            seed:        随机种子
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split

        img_dir = self.root_dir / "images"
        ann_dir = self.root_dir / "annotations"

        # ---- 自动检测目录结构 ----
        # 情况 A: images/training/ + images/test/ (你的数据集格式)
        # 情况 B: images/ 下直接是图片文件 (扁平结构)
        has_subfolders = (img_dir / "training").is_dir() or (img_dir / "train").is_dir()

        if has_subfolders:
            # 确定子目录名 (training 或 train)
            train_sub = "training" if (img_dir / "training").is_dir() else "train"
            test_sub = "test" if (img_dir / "test").is_dir() else "val"

            if split == "train":
                img_search_dir = img_dir / train_sub
                ann_search_dir = ann_dir / train_sub
            else:  # val
                img_search_dir = img_dir / test_sub
                ann_search_dir = ann_dir / test_sub

            print(f"[NEU-Seg] 检测到子目录结构: images/{train_sub}/, images/{test_sub}/")
        else:
            img_search_dir = img_dir
            ann_search_dir = ann_dir

        # ---- 查找所有图像 (递归搜索) ----
        img_files = sorted(
            glob.glob(str(img_search_dir / "**" / "*.jpg"), recursive=True)
            + glob.glob(str(img_search_dir / "**" / "*.bmp"), recursive=True)
            + glob.glob(str(img_search_dir / "**" / "*.png"), recursive=True)
            + glob.glob(str(img_search_dir / "*.jpg"))
            + glob.glob(str(img_search_dir / "*.bmp"))
            + glob.glob(str(img_search_dir / "*.png"))
        )
        # 去重
        img_files = sorted(set(img_files))

        # ---- 构建 (image, mask) 对 ----
        self.pairs = []
        for img_path in img_files:
            stem = Path(img_path).stem
            # 在对应的 annotation 目录中查找 mask
            for ext in [".png", ".bmp", ".jpg"]:
                mask_path = ann_search_dir / f"{stem}{ext}"
                if mask_path.exists():
                    self.pairs.append((str(img_path), str(mask_path)))
                    break

        assert len(self.pairs) > 0, (
            f"未找到数据! 请确认目录结构:\n"
            f"  搜索图像目录: {img_search_dir}/\n"
            f"  搜索标注目录: {ann_search_dir}/\n"
            f"  找到图像文件: {len(img_files)} 张\n"
            f"  匹配的标注对: {len(self.pairs)} 对"
        )

        # ---- 划分训练/验证集 ----
        if has_subfolders:
            # 如果数据集本身就有 training/test 划分, 直接使用全部数据
            self.indices = list(range(len(self.pairs)))
        else:
            # 扁平结构, 手动按比例划分
            random.seed(seed)
            indices = list(range(len(self.pairs)))
            random.shuffle(indices)
            split_idx = int(len(indices) * train_ratio)
            self.indices = indices[:split_idx] if split == "train" else indices[split_idx:]

        print(f"[NEU-Seg] {split}: {len(self.indices)} 张图像, "
              f"共 {len(self.pairs)} 张, 缺陷类别: {self.CLASS_NAMES[1:]}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, mask_path = self.pairs[real_idx]

        # 读取图像 (灰度 → RGB)
        img = Image.open(img_path)
        if img.mode == "L":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)

        # 读取 mask
        mask = np.array(Image.open(mask_path))

        # 如果 mask 是 RGB, 转为类别索引
        if mask.ndim == 3:
            mask = self._rgb_mask_to_class(mask)

        # 确保 mask 值在合理范围
        mask = np.clip(mask, 0, self.NUM_CLASSES - 1).astype(np.int64)

        # Resize 到 SAM2 输入尺寸
        img, mask = self._resize(img, mask)

        # 生成随机 point prompt (从缺陷区域采样)
        point, label = self._sample_point_prompt(mask)

        # 转为 tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return {
            "image": img_tensor,            # (3, H, W)
            "mask": mask_tensor,            # (H, W) 类别索引
            "point_coords": point,          # (1, 2) 采样点坐标
            "point_labels": label,          # (1,)  1=前景
            "image_path": img_path,
        }

    def _resize(self, img, mask):
        """Resize image and mask to target size"""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
        )

        # Pad to img_size x img_size
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask_padded = np.pad(mask_resized, ((0, pad_h), (0, pad_w)), mode="constant")

        return img_padded, mask_padded

    def _sample_point_prompt(self, mask):
        """从缺陷区域中随机采样一个点作为 prompt"""
        defect_mask = (mask > 0)
        if defect_mask.sum() > 0:
            ys, xs = np.where(defect_mask)
            idx = random.randint(0, len(ys) - 1)
            point = np.array([[xs[idx], ys[idx]]], dtype=np.float32)
            label = np.array([1], dtype=np.int64)  # 前景点
        else:
            # 如果没有缺陷, 随机采样一个背景点
            h, w = mask.shape
            point = np.array([[w // 2, h // 2]], dtype=np.float32)
            label = np.array([0], dtype=np.int64)  # 背景点
        return point, label

    def _rgb_mask_to_class(self, rgb_mask):
        """将 RGB 格式的 mask 转换为类别索引"""
        # 常见的 NEU-Seg 颜色编码
        # patches=蓝(0,0,255)=1, inclusion=绿(0,255,0)=2, scratches=红(255,0,0)=3
        class_mask = np.zeros(rgb_mask.shape[:2], dtype=np.int64)
        if rgb_mask.ndim == 3:
            class_mask[(rgb_mask[:, :, 2] > 128) & (rgb_mask[:, :, 0] < 128)] = 1  # 蓝
            class_mask[(rgb_mask[:, :, 1] > 128) & (rgb_mask[:, :, 0] < 128) & (rgb_mask[:, :, 2] < 128)] = 2  # 绿
            class_mask[(rgb_mask[:, :, 0] > 128) & (rgb_mask[:, :, 1] < 128)] = 3  # 红
        return class_mask


# ============================================================
#  Severstal 数据集 (Kaggle)
#  4类缺陷: Class1-4
#  图像: 1600x256
# ============================================================
class SeverstalDataset(Dataset):
    """Severstal Steel Defect Detection (Kaggle) 分割数据集"""

    CLASS_NAMES = ["background", "class1", "class2", "class3", "class4"]
    NUM_CLASSES = 5

    def __init__(self, root_dir, split="train", img_size=1024, train_ratio=0.8, seed=42):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split

        csv_path = self.root_dir / "train.csv"
        assert csv_path.exists(), f"找不到 {csv_path}, 请确认 Severstal 数据已下载"

        df = pd.read_csv(csv_path)
        # Kaggle 两种 CSV 格式:
        #   旧: ImageId_ClassId = "xxxxx.jpg_1"
        #   新: 独立列 ImageId, ClassId, EncodedPixels
        if "ImageId_ClassId" in df.columns:
            df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
            df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: int(x.split("_")[1]))
        elif {"ImageId", "ClassId"}.issubset(df.columns):
            df["ClassId"] = df["ClassId"].astype(int)
        else:
            raise ValueError(f"{csv_path} 列格式未知, 需要 ImageId+ClassId 或 ImageId_ClassId")

        # 只保留有缺陷的图像
        defect_df = df.dropna(subset=["EncodedPixels"])
        self.image_ids = sorted(defect_df["ImageId"].unique().tolist())
        self.df = df

        # 划分
        random.seed(seed)
        indices = list(range(len(self.image_ids)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        self.indices = indices[:split_idx] if split == "train" else indices[split_idx:]

        print(f"[Severstal] {split}: {len(self.indices)} 张有缺陷图像")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_id = self.image_ids[real_idx]

        # 读取图像
        img_path = self.root_dir / "train_images" / image_id
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # 解码 RLE mask
        mask = np.zeros((h, w), dtype=np.int64)
        for cls_id in range(1, 5):
            rle_row = self.df[
                (self.df["ImageId"] == image_id) & (self.df["ClassId"] == cls_id)
            ]
            if len(rle_row) > 0 and pd.notna(rle_row.iloc[0]["EncodedPixels"]):
                rle = rle_row.iloc[0]["EncodedPixels"]
                cls_mask = self._rle_decode(rle, (h, w))
                mask[cls_mask > 0] = cls_id

        # Resize
        img, mask = self._resize(img, mask)

        # Sample prompt
        point, label = self._sample_point_prompt(mask)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "point_coords": point,
            "point_labels": label,
            "image_path": str(img_path),
        }

    def _resize(self, img, mask):
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
        )
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask_padded = np.pad(mask_resized, ((0, pad_h), (0, pad_w)), mode="constant")
        return img_padded, mask_padded

    def _sample_point_prompt(self, mask):
        defect_mask = (mask > 0)
        if defect_mask.sum() > 0:
            ys, xs = np.where(defect_mask)
            idx = random.randint(0, len(ys) - 1)
            return np.array([[xs[idx], ys[idx]]], dtype=np.float32), np.array([1], dtype=np.int64)
        h, w = mask.shape
        return np.array([[w // 2, h // 2]], dtype=np.float32), np.array([0], dtype=np.int64)

    @staticmethod
    def _rle_decode(rle_str, shape):
        """Decode Kaggle RLE format"""
        s = list(map(int, rle_str.split()))
        starts, lengths = s[0::2], s[1::2]
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, length in zip(starts, lengths):
            mask[start - 1: start - 1 + length] = 1
        return mask.reshape(shape, order="F")  # Fortran order (column-major)


# ============================================================
#  工厂函数
# ============================================================
def get_dataset(name, root_dir, split="train", img_size=1024):
    """根据数据集名称返回对应的 Dataset"""
    datasets = {
        "neu_seg": NEUSegDataset,
        "severstal": SeverstalDataset,
    }
    assert name in datasets, f"不支持的数据集: {name}, 可选: {list(datasets.keys())}"
    return datasets[name](root_dir=root_dir, split=split, img_size=img_size)


def _seed_worker(worker_id):
    """DataLoader worker 的可复现随机种子初始化."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size=2, shuffle=True, num_workers=4, seed=None):
    """
    seed 非空时: 固定 shuffle 顺序 + worker 随机性, 保证跨运行可复现.
    """
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(True if dataset.split == "train" else False),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


# ============================================================
#  测试
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="neu_seg", choices=["neu_seg", "severstal"])
    parser.add_argument("--data_dir", default="data/NEU-Seg")
    args = parser.parse_args()

    ds = get_dataset(args.dataset, args.data_dir, split="train")
    sample = ds[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape:  {sample['mask'].shape}")
    print(f"Mask classes: {torch.unique(sample['mask']).tolist()}")
    print(f"Point coord:  {sample['point_coords']}")
    print(f"Point label:  {sample['point_labels']}")
    print(f"✅ 数据集加载成功!")
