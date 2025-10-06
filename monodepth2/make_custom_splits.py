import os
import glob
import random

# 假设你的图片都放在这个目录下
data_dir = "/content/drive/MyDrive/my_data"

# 找出所有图片
all_images = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

print(f"总共找到 {len(all_images)} 张图片")

# 打乱顺序（保证不同视频帧混合）
random.shuffle(all_images)

# 划分比例
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

n = len(all_images)
train_files = all_images[:int(n*train_ratio)]
val_files   = all_images[int(n*train_ratio):int(n*(train_ratio+val_ratio))]
test_files  = all_images[int(n*(train_ratio+val_ratio)):]

# 写入 splits/custom/
os.makedirs("splits/custom", exist_ok=True)

def write_split(file_list, filename):
    with open(filename, "w") as f:
        for path in file_list:
            # 转换成相对路径（相对 data_path）
            rel_path = os.path.relpath(path, "/content/drive/MyDrive")
            f.write(rel_path + "\n")

write_split(train_files, "splits/custom/train_files.txt")
write_split(val_files,   "splits/custom/val_files.txt")
write_split(test_files,  "splits/custom/test_files.txt")

print("train/val/test 文件已生成到 splits/custom/")
