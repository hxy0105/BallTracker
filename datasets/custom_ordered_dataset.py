# datasets/custom_ordered_dataset.py
import os
import re
from collections import defaultdict
from PIL import Image
import random

from .mono_dataset import MonoDataset
from torchvision import transforms
from torchvision.transforms import functional as F


class CustomOrderedDataset(MonoDataset):
    """
    使用你在 splits/<name>/*.txt 里给出的“相对 data_path 的单列文件路径”，
    且这些路径已经按时间顺序排列。
    - 自动按“视频名”（basename 去掉 `_frame_xxx`）分组
    - 取样只在同一视频内做 [-1, 0, +1]，不会跨视频
    - 组内严格保持原顺序
    """

    def __init__(self, *args, strict_neighbors=False, **kwargs):
        """
        strict_neighbors=True 时：如果某个样本在视频边界处，导致 -1 或 +1 不存在，则跳过该样本（通过构造 sample 索引避免边界）。
        strict_neighbors=False（默认）：边界处邻帧用“就近夹取”（clamp），即用边界帧自己代替。
        """
        super(CustomOrderedDataset, self).__init__(*args, **kwargs)
        self.strict_neighbors = strict_neighbors

        # 读取 self.filenames（父类已读 txt 得到列表：每行一条相对路径）
        # 规范化为统一的 '/' 分隔符
        self.files = [p.strip().replace("\\", "/") for p in self.filenames]
        # 过滤空行
        self.files = [p for p in self.files if p]

        # 按视频分组：视频 key = 去掉文件名中的 `_frame_数字` 部分
        self._frame_re = re.compile(r"_frame_(\d+)")
        groups = defaultdict(list)
        for rel in self.files:
            bn = os.path.basename(rel)
            vid = bn.split("_frame")[0]  # 例：xxx.mp4
            groups[vid].append(rel)

        # 组内按帧号（_frame_####）排序，保证顺序
        def frame_idx(rel_path: str):
            m = self._frame_re.search(os.path.basename(rel_path))
            return int(m.group(1)) if m else -1

        self.videos = {}  # vid -> 有序相对路径列表
        for vid, lst in groups.items():
            lst = sorted(lst, key=frame_idx)
            self.videos[vid] = lst

        # 构建“可训练样本索引”：每个元素 = (vid, pos)
        self.samples = []
        for vid, seq in self.videos.items():
            L = len(seq)
            if L == 0:
                continue
            if self.strict_neighbors and any(i in (-1, 1) for i in self.frame_ids):
                # 需要邻帧且严格要求存在：去掉两端
                left = max(0, 0 - min(self.frame_ids))     # 通常=1
                right = L - max(self.frame_ids)            # 通常=L-1
                for pos in range(left, right):
                    self.samples.append((vid, pos))
            else:
                # 不严格：全保留，邻帧超界时会做 clamp
                for pos in range(L):
                    self.samples.append((vid, pos))

        # 覆盖父类默认长度
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    # 本数据集没有 KITTI 的 GT 深度
    def check_depth(self):
        return False

    # —— 核心：根据 (vid, pos) + frame_ids 取同视频的邻帧 ——
    def _path_at(self, vid: str, pos: int):
        """返回 data_path 下的绝对路径"""
        rel = self.videos[vid][pos]
        return os.path.join(self.data_path, rel)

    def get_color(self, folder, frame_index, side, do_flip):
        # 这里不会被直接调用（我们在 __getitem__ 自己管路径），但保持接口需要
        raise NotImplementedError

    def __getitem__(self, index):
        inputs = {}

        # 采样索引（视频 id + 在该视频中的位置）
        vid, pos = self.samples[index]

        # 数据增强标志（沿用父类逻辑）
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # 为每个需要的相对帧 i in frame_ids 准备图像
        seq_len = len(self.videos[vid])
        for i in self.frame_idxs:
            if self.strict_neighbors:
                # 严格模式下样本已保证不越界
                p = pos + i
            else:
                # 非严格：越界时 clamp 在 [0, seq_len-1]
                p = min(max(pos + i, 0), seq_len - 1)
            img_path = self._path_at(vid, p)

            color = Image.open(img_path).convert('RGB')
            if do_flip:
                color = color.transpose(Image.FLIP_LEFT_RIGHT)

            # 与父类保持一致的 key 约定：("color", i, s)
            inputs[("color", i, -1)] = color

        # 预处理（金字塔、多尺度、张量化、颜色增强）
        if do_color_aug:
            params = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
                # 兼容不同版本：有的返回函数，有的返回(b, c, s, h)
            if isinstance(params, tuple):
                b, c, s, h = params
                def color_aug(img):
                    img = F.adjust_brightness(img, b)
                    img = F.adjust_contrast(img, c)
                    img = F.adjust_saturation(img, s)
                    img = F.adjust_hue(img, h)
                    return img
            else:
                color_aug = params  # 已经是可调用
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        #self.preprocess(inputs, do_color_aug, do_flip)

        # 相机内参：沿用父类的约定
        for i in self.frame_idxs:
            inputs[("color_aug", i, 0)] = inputs[("color", i, 0)]

        # 其他必须字段（保持兼容）
        inputs["index"] = index
        return inputs
