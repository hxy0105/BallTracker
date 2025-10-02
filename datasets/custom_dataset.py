from .mono_dataset import MonoDataset
from PIL import Image
import os

class CustomSeqDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CustomSeqDataset, self).__init__(*args, **kwargs)
        # 这里假设 split 文件已经是顺序排列的
        with open(self.filenames_path, "r") as f:
            self.image_files = [line.strip() for line in f]

    def check_depth(self):
        return False

    def get_image_path(self, idx):
        return os.path.join(self.data_path, self.image_files[idx])

    def get_color(self, folder, frame_index, side, do_flip):
        # 这里直接用 idx + frame_index 来取
        img_path = self.get_image_path(frame_index)
        color = Image.open(img_path).convert('RGB')
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color
