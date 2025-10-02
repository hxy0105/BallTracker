import os, glob, random, re
from collections import defaultdict

# ========= 修改这里 =========
base_path = "E:/balltracker/ball_tracker"   # 训练时 --data_path 也用它
data_dir  = os.path.join(base_path, "images", "dataset_1000_to_tag")
flag_dir  = os.path.join(base_path, "labels", "data", "3x3", "labels", "final_extraction", "all_annotations", "flag")
# ===========================

seed = 2025
random.seed(seed)

# 1) 收集所有图片，统一路径分隔符
all_images = glob.glob(os.path.join(data_dir, "*.png"))
all_images = [os.path.normpath(p).replace("\\", "/") for p in all_images]  # 统一成 /
print(f"[Info] 初始找到 {len(all_images)} 张图片")

# 去重
all_images = sorted(set(all_images))
print(f"[Info] 去重后 {len(all_images)} 张图片")

# 2) 按视频分组
def video_key(path: str):
    base = os.path.basename(path)
    return base.split("_frame")[0]  # 提取视频名（不含帧号）

frame_num_re = re.compile(r"_frame_(\d+)")
def frame_index(path: str):
    m = frame_num_re.search(os.path.basename(path))
    return int(m.group(1)) if m else -1

groups = defaultdict(list)
for p in all_images:
    groups[video_key(p)].append(p)

# 组内按帧号排序
for k in groups:
    groups[k] = sorted(groups[k], key=frame_index)

video_keys = list(groups.keys())
print(f"[Info] 视频数量: {len(video_keys)}")

# 3) 打乱视频顺序
random.shuffle(video_keys)

# 4) 按视频划分 train/val/test（不拆视频）
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
n_vids = len(video_keys)
n_train = int(n_vids * train_ratio)
n_val   = int(n_vids * val_ratio)
n_test  = n_vids - n_train - n_val

train_videos = video_keys[:n_train]
val_videos   = video_keys[n_train:n_train+n_val]
test_videos  = video_keys[n_train+n_val:]

def flatten(vkeys):
    out = []
    for v in vkeys:
        out.extend(groups[v])
    return out

train_files = flatten(train_videos)
val_files   = flatten(val_videos)
test_files  = flatten(test_videos)

print(f"[Split] videos -> train={len(train_videos)}, val={len(val_videos)}, test={len(test_videos)}")
print(f"[Split] frames -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

# 5) 保存
output_dir = os.path.join(base_path, "splits", "custom")
os.makedirs(output_dir, exist_ok=True)

def write_split(file_list, filename):
    seen = set()
    with open(filename, "w") as f:
        for p in file_list:
            if p not in seen:  # 避免重复
                rel_path = os.path.relpath(p, base_path).replace("\\", "/")
                f.write(rel_path + "\n")
                seen.add(p)

write_split(train_files, os.path.join(output_dir, "train_files.txt"))
write_split(val_files,   os.path.join(output_dir, "val_files.txt"))
write_split(test_files,  os.path.join(output_dir, "test_files.txt"))

print("[Done] 已生成 splits/custom/train_files.txt, val_files.txt, test_files.txt")
