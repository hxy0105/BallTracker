import os
import subprocess
from pathlib import Path

# === 配置 ===
IMG_DIR = r"E:\balltracker\ball_tracker\images\dataset_1000_to_tag"
MODEL_NAME = "mono_640x192"  # 也可改为 midas_v21_small 等
PY_FILE = r"E:\balltracker\ball_tracker\monodepth2\monodepth2\test_simple.py"
OVERWRITE = False  # 若为 True，则覆盖已有深度图；False 则跳过

# === 主流程 ===
img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
img_paths = [p for p in Path(IMG_DIR).rglob("*") if p.suffix.lower() in img_exts]

print(f"[I] 检测到 {len(img_paths)} 张图片，将逐一生成深度图...")

for img_path in img_paths:
    disp_path = img_path.with_name(img_path.stem + "_disp.jpeg")

    if disp_path.exists() and not OVERWRITE:
        print(f"[→] 已存在: {disp_path.name}，跳过。")
        continue

    print(f"[+] 处理: {img_path.name}")
    cmd = [
        "python", PY_FILE,
        "--image_path", str(img_path),
        "--model_name", MODEL_NAME
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[X] 处理失败: {img_path.name}, 错误: {e}")

print("\n✅ 全部完成。")
