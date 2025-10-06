import cv2
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple

# =========================
# 配置区（按需修改）
# =========================
W, H = 1920, 1080
FOLDER_2D = r"E:\balltracker\ball_tracker\labels\data\3x3\labels\final_extraction\all_annotations\court_keypoints_yolo_pose"
NORMALIZED_2D = True                 # 文件里是0~1坐标
START_FROM_THIRD_3D = False          # 若想从第3个3D点开始对齐(16点)，改为 True
TOKENS_SKIP_HEAD = 5                 # 每个文件前面需要跳过的token数（你给的是5）

# 随机抽样参数
N_STAGE1 = 20                        # 阶段一帧数
N_STAGE2 = 80                        # 阶段二帧数
RANDOM_SEED = 42                     # None 表示每次不同；设数值可复现
STAGE2_SUPERSET = True               # 阶段二是否包含阶段一再扩充（但限定在 all2 内）

# 优化/剔除设置
ERR_THRESH_PX = 4.0               # 逐帧误差剔除阈值（像素）
USE_RATIONAL = False                 # 是否启用 k4~k6（广角再开）
SORT_FILES = False                   # 文件已按顺序命名则 False

# 你的18个3D参考点（Z=0）
PTS3D_ALL = np.array([
    [0.0,  0.0, 0.0],
    [0.9,  0.0, 0.0],
    [5.05, 0.0, 0.0],
    [5.05, 1.75,0.0],
    [5.05, 2.6, 0.0],
    [5.05, 3.0, 0.0],
    [5.05, 3.85,0.0],
    [5.05, 4.7, 0.0],
    [5.05, 5.8, 0.0],
    [9.95, 5.8, 0.0],
    [9.95, 4.7, 0.0],
    [9.95, 3.85,0.0],
    [9.95, 3.0, 0.0],
    [9.95, 2.6, 0.0],
    [9.95, 1.75,0.0],
    [9.95, 0.0, 0.0],
    [14.1, 0.0, 0.0],
    [15.0, 0.0, 0.0],
    #[7.50, 1.575, 3.05]
], dtype=np.float32)

PTS3D_REF = PTS3D_ALL[2:].copy() if START_FROM_THIRD_3D else PTS3D_ALL.copy()

# —— 新增：篮筐 bbox 配置 ——
HOOP_FOLDER = r"E:\balltracker\ball_tracker\labels\data\3x3\labels\final_extraction\all_annotations\rim"   # 放 bbox 的txt文件夹
HOOP_NORMALIZED = True                             # 你的bbox是0~1归一化
HOOP_TOKENS_SKIP_HEAD = 0                          # 若文件前需要跳过token，可调
HOOP_USE = False                                    # 是否启用篮筐点
HOOP3D = (7.50, 1.575, 3.05)                       # 篮筐3D圆心(单位：米)


# =========================
# 工具函数
# =========================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_tokens_from_file(fp: Path):
    txt = fp.read_text(encoding="utf-8", errors="ignore").replace(",", " ")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    tokens = []
    for ln in lines:
        tokens += ln.split()
    return tokens

def parse_uv_and_labs(tokens):
    """
    解析三元组 (u, v, label)，
    返回:
      uv:   (N,2) float32  -> 像素坐标 (u,v)
      labs: (N,)  int32    -> 可见性/标签
    """
    assert len(tokens) % 3 == 0, "文件里的数字不是3的倍数，无法按 (u,v,label) 解析"
    uv_list, labs = [], []
    for i in range(0, len(tokens), 3):
        u, v = float(tokens[i]), float(tokens[i+1])
        lab = int(float(tokens[i+2]))
        uv_list.append([u, v])
        labs.append(lab)
    return np.array(uv_list, dtype=np.float32), np.array(labs, dtype=np.int32)


# def load_folder_uv_and_mask(folder_2d: str, normalized: bool, W: int, H: int, skip_head: int):
#     folder = Path(folder_2d)
#     if SORT_FILES:
#         files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".txt", ".csv")],
#                        key=lambda p: natural_key(p.name))
#     else:
#         files = [p for p in folder.iterdir() if p.suffix.lower() in (".txt", ".csv")]
#     if not files:
#         raise FileNotFoundError(f"在 {folder} 未找到 .txt/.csv 文件")

#     uv_list, names, all2_mask = [], [], []
#     for fp in files:
#         tokens = read_tokens_from_file(fp)
#         if skip_head > 0:
#             tokens = tokens[skip_head:]
#         uv, labs = parse_uv_and_labs(tokens)
#         if normalized:
#             uv[:, 0] *= W
#             uv[:, 1] *= H
#         uv_list.append(uv)
#         names.append(fp.name)
#         all2_mask.append(bool(np.all(labs == 2)))
#     return uv_list, names, all2_mask

def load_folder_uv_and_labs(folder_2d: str, normalized: bool, W: int, H: int, skip_head: int):
    folder = Path(folder_2d)
    if SORT_FILES:
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".txt", ".csv")],
                       key=lambda p: natural_key(p.name))
    else:
        files = [p for p in folder.iterdir() if p.suffix.lower() in (".txt", ".csv")]
    if not files:
        raise FileNotFoundError(f"在 {folder} 未找到 .txt/.csv 文件")

    uv_list, labs_list, names = [], [], []
    for fp in files:
        tokens = read_tokens_from_file(fp)
        if skip_head > 0:
            tokens = tokens[skip_head:]
        uv, labs = parse_uv_and_labs(tokens)
        if normalized:
            uv[:, 0] *= W
            uv[:, 1] *= H
        uv_list.append(uv)
        labs_list.append(labs)
        names.append(fp.name)
    return uv_list, labs_list, names


def build_correspondences_per_frame(uv_list, labs_list, pts3d_ref, min_pts=6):
    objectPoints, imagePoints = [], []
    for uv, labs in zip(uv_list, labs_list):
        mask = (labs == 2)             # 只保留可见点
        if mask.sum() < min_pts:       # 至少要有 min_pts 个点才能PnP
            continue
        objectPoints.append(pts3d_ref[mask].astype(np.float32))
        imagePoints.append(uv[mask].reshape(-1, 1, 2).astype(np.float32))
    return objectPoints, imagePoints


def build_correspondences(uv_list, pts3d_ref):
    objectPoints, imagePoints = [], []
    for uv in uv_list:
        if uv.shape[0] != pts3d_ref.shape[0]:
            raise ValueError(f"2D点数({uv.shape[0]}) != 3D点数({pts3d_ref.shape[0]})")
        objectPoints.append(pts3d_ref.astype(np.float32))
        imagePoints.append(uv.reshape(-1, 1, 2).astype(np.float32))
    return objectPoints, imagePoints

def calibrate(objectPoints, imagePoints, imageSize, K_init=None, dist_init=None, use_guess=False):
    flags = 0
    if use_guess:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    if USE_RATIONAL:
        flags |= cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-7)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, imageSize,
        K_init, dist_init, flags=flags, criteria=criteria
    )
    return rms, K, dist, rvecs, tvecs

def per_frame_error(objectPoints, imagePoints, K, dist, rvecs, tvecs):
    errs = []
    for i,(P3,P2) in enumerate(zip(objectPoints, imagePoints)):
        proj, _ = cv2.projectPoints(P3, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(P2, proj, cv2.NORM_L2) / len(P3)
        errs.append(float(err))
    return np.array(errs, dtype=np.float32)

def prune_by_error(objectPoints, imagePoints, names, errs, thr_px=2.0):
    keep = [i for i,e in enumerate(errs) if e <= thr_px]
    return [objectPoints[i] for i in keep], [imagePoints[i] for i in keep], [names[i] for i in keep], keep

# —— 新增：读取篮筐bbox中心（像素坐标），以“文件名不含扩展名”为键 —— 
def load_hoop_centers(folder_hoop: str, normalized: bool, W: int, H: int, skip_head: int):
    """
    返回 dict: { 文件stem(str) : (u,v) 像素坐标 }
    期望每个txt为一行或多行，格式： 0 cx cy w h  （归一化或像素，按normalized决定）
    若一文件多行，默认取第1行（你也可以改成取最大面积行）
    """
    folder = Path(folder_hoop)
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".txt", ".csv")]
    centers = {}
    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            continue
        # 只取第一行（若你要取最大 w*h 的那行，可自行排序选择）
        line = txt.splitlines()[0].strip().replace(",", " ")
        toks = [t for t in line.split() if t]
        if skip_head > 0:
            toks = toks[skip_head:]
        if len(toks) < 5:
            # 兼容形如 "0 cx cy w h"
            continue
        # 解析：0 cx cy w h
        _lab = float(toks[0])
        cx = float(toks[1]); cy = float(toks[2])
        w_norm = float(toks[3]); h_norm = float(toks[4])
        if normalized:
            u = cx * W
            v = (cy-h_norm) * H
        else:
            u = cx
            v = cy
        centers[fp.stem] = (float(u), float(v))
    return centers


# —— 新增：逐帧构建对应关系（仅用 lab=2 的地面点 + 可选篮筐中心点） —— 
def build_correspondences_with_hoop(uv_list, labs_list, names, pts3d_ref,
                                    hoop_centers=None, use_hoop=True, hoop3d=(7.5,1.575,3.05), min_pts=6):
    """
    按帧构建：
      - 地面点：只取 lab==2 的 (u,v) ↔ 对应 PTS3D_REF 的相同索引（假设顺序已对齐）
      - 篮筐点：若 use_hoop 且能在 hoop_centers 中找到同名（按stem匹配），则附加 (u,v) ↔ hoop3d
    要求每帧总点数 >= min_pts
    """
    objectPoints, imagePoints = [], []
    for uv, labs, name in zip(uv_list, labs_list, names):
        mask = (labs == 2)
        P3 = pts3d_ref[mask]
        P2 = uv[mask]
        # 加入篮筐中心（若有）
        if use_hoop and hoop_centers is not None:
            stem = Path(name).stem
            if stem in hoop_centers:
                u_h, v_h = hoop_centers[stem]
                P3 = np.vstack([P3, np.array(hoop3d, dtype=np.float32)])
                P2 = np.vstack([P2, np.array([u_h, v_h], dtype=np.float32)])
        # 至少 min_pts
        if P3.shape[0] >= min_pts:
            objectPoints.append(P3.astype(np.float32))
            imagePoints.append(P2.reshape(-1,1,2).astype(np.float32))
    return objectPoints, imagePoints

# =========================
# 主流程
# =========================
# def main():
#     uv_list, names, all2_mask = load_folder_uv_and_mask(FOLDER_2D, NORMALIZED_2D, W, H, TOKENS_SKIP_HEAD)
#     objectPoints_all, imagePoints_all = build_correspondences(uv_list, PTS3D_REF)
#     N = len(objectPoints_all)
#     print(f"[I] 读取到 {N} 帧，每帧 {PTS3D_REF.shape[0]} 个点")
#     idx_all2 = [i for i, ok in enumerate(all2_mask) if ok]
#     print(f"[I] 满足 “该帧所有 lab=2” 的候选帧数：{len(idx_all2)}")

#     if len(idx_all2) == 0:
#         raise RuntimeError("没有任何一帧满足 ‘lab 全为 2’，无法标定。")

#     rng = np.random.default_rng(RANDOM_SEED)
#     # 阶段一
#     n1 = min(N_STAGE1, len(idx_all2))
#     idx_stage1 = rng.choice(idx_all2, size=n1, replace=False).tolist()
#     # 阶段二
#     n2 = min(N_STAGE2, len(idx_all2))
#     if STAGE2_SUPERSET:
#         if n2 <= n1:
#             idx_stage2 = idx_stage1[:n2]
#         else:
#             remain = [i for i in idx_all2 if i not in idx_stage1]
#             add = rng.choice(len(remain), size=(n2 - n1), replace=False)
#             idx_stage2 = idx_stage1 + [remain[j] for j in add]
#     else:
#         idx_stage2 = rng.choice(idx_all2, size=n2, replace=False).tolist()

#     # 阶段一
#     obj_s1 = [objectPoints_all[i] for i in idx_stage1]
#     img_s1 = [imagePoints_all[i] for i in idx_stage1]
#     print(f"[I] 阶段一：在 lab=2 的 {len(idx_all2)} 帧中随机取 {len(obj_s1)} 帧做初标定...")
#     rms1, K1, dist1, rvecs1, tvecs1 = calibrate(obj_s1, img_s1, (W,H), None, None, use_guess=False)
#     print(f"[I] 阶段一 RMS = {rms1:.3f}px\nK=\n{K1}\ndist={dist1.ravel()}")

#     # 阶段二
#     obj_s2 = [objectPoints_all[i] for i in idx_stage2]
#     img_s2 = [imagePoints_all[i] for i in idx_stage2]
#     names_s2 = [names[i] for i in idx_stage2]
#     print(f"[I] 阶段二：在 lab=2 的 {len(idx_all2)} 帧中随机取 {len(obj_s2)} 帧精炼...")
#     rms2, K2, dist2, rvecs2, tvecs2 = calibrate(obj_s2, img_s2, (W,H), K1, dist1, use_guess=True)
#     errs2 = per_frame_error(obj_s2, img_s2, K2, dist2, rvecs2, tvecs2)
#     print(f"[I] 阶段二 RMS = {rms2:.3f}px，mean={errs2.mean():.3f}, 95%={np.percentile(errs2,95):.3f}")

#     # 误差剔除 → 最终
#     obj_keep, img_keep, names_keep, keep_idx = prune_by_error(obj_s2, img_s2, names_s2, errs2, thr_px=ERR_THRESH_PX)
#     print(f"[I] 剔除阈值 {ERR_THRESH_PX}px 后保留 {len(obj_keep)}/{len(obj_s2)} 帧")
#     rmsF, KF, distF, rvecsF, tvecsF = calibrate(obj_keep, img_keep, (W,H), K2, dist2, use_guess=True)
#     errsF = per_frame_error(obj_keep, img_keep, KF, distF, rvecsF, tvecsF)
#     print(f"[I] 最终 RMS = {rmsF:.3f}px，mean={errsF.mean():.3f}, 95%={np.percentile(errsF,95):.3f}")
#     print("[I] 最终内参 K =\n", KF)
#     print("[I] 最终畸变 dist = ", distF.ravel())

#     out = Path("camera_calib.yaml")
#     fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
#     fs.write("K", KF); fs.write("dist", distF)
#     fs.write("image_width", int(W)); fs.write("image_height", int(H))
#     fs.write("rms", float(rmsF))
#     fs.release()
#     print(f"[I] 已保存到 {out.resolve()}")

# def main():
#     uv_list, labs_list, names = load_folder_uv_and_labs(FOLDER_2D, NORMALIZED_2D, W, H, TOKENS_SKIP_HEAD)
#     objectPoints_all, imagePoints_all = build_correspondences_per_frame(uv_list, labs_list, PTS3D_REF)

#     N = len(objectPoints_all)
#     print(f"[I] 读取到 {len(uv_list)} 帧，其中可用 {N} 帧 (至少有足够的 lab=2 点)")

#     if N == 0:
#         raise RuntimeError("没有任何一帧有足够的可见点，无法标定。")

#     rng = np.random.default_rng(RANDOM_SEED)
#     # 阶段一
#     n1 = min(N_STAGE1, N)
#     idx_stage1 = rng.choice(N, size=n1, replace=False).tolist()
#     # 阶段二
#     n2 = min(N_STAGE2, N)
#     if STAGE2_SUPERSET:
#         if n2 <= n1:
#             idx_stage2 = idx_stage1[:n2]
#         else:
#             remain = [i for i in range(N) if i not in idx_stage1]
#             add = rng.choice(len(remain), size=(n2 - n1), replace=False)
#             idx_stage2 = idx_stage1 + [remain[j] for j in add]
#     else:
#         idx_stage2 = rng.choice(N, size=n2, replace=False).tolist()

#     # 阶段一
#     obj_s1 = [objectPoints_all[i] for i in idx_stage1]
#     img_s1 = [imagePoints_all[i] for i in idx_stage1]
#     print(f"[I] 阶段一：随机取 {len(obj_s1)} 帧做初标定...")
#     rms1, K1, dist1, rvecs1, tvecs1 = calibrate(obj_s1, img_s1, (W,H), None, None, use_guess=False)
#     print(f"[I] 阶段一 RMS = {rms1:.3f}px\nK=\n{K1}\ndist={dist1.ravel()}")

#     # 阶段二
#     obj_s2 = [objectPoints_all[i] for i in idx_stage2]
#     img_s2 = [imagePoints_all[i] for i in idx_stage2]
#     names_s2 = [names[i] for i in idx_stage2]
#     print(f"[I] 阶段二：随机取 {len(obj_s2)} 帧精炼...")
#     rms2, K2, dist2, rvecs2, tvecs2 = calibrate(obj_s2, img_s2, (W,H), K1, dist1, use_guess=True)
#     errs2 = per_frame_error(obj_s2, img_s2, K2, dist2, rvecs2, tvecs2)
#     print(f"[I] 阶段二 RMS = {rms2:.3f}px，mean={errs2.mean():.3f}, 95%={np.percentile(errs2,95):.3f}")

#     # 误差剔除 → 最终
#     obj_keep, img_keep, names_keep, keep_idx = prune_by_error(obj_s2, img_s2, names_s2, errs2, thr_px=ERR_THRESH_PX)
#     print(f"[I] 剔除阈值 {ERR_THRESH_PX}px 后保留 {len(obj_keep)}/{len(obj_s2)} 帧")
#     rmsF, KF, distF, rvecsF, tvecsF = calibrate(obj_keep, img_keep, (W,H), K2, dist2, use_guess=True)
#     errsF = per_frame_error(obj_keep, img_keep, KF, distF, rvecsF, tvecsF)
#     print(f"[I] 最终 RMS = {rmsF:.3f}px，mean={errsF.mean():.3f}, 95%={np.percentile(errsF,95):.3f}")
#     print("[I] 最终内参 K =\n", KF)
#     print("[I] 最终畸变 dist = ", distF.ravel())

def main():
    # 读取 2D 点 + labs
    uv_list, labs_list, names = load_folder_uv_and_labs(FOLDER_2D, NORMALIZED_2D, W, H, TOKENS_SKIP_HEAD)
    print(f"[I] 读取到 {len(uv_list)} 帧（关键点）")

    # 读取篮筐 bbox → 2D 中心点（像素）
    hoop_centers = None
    if HOOP_USE:
        hoop_centers = load_hoop_centers(HOOP_FOLDER, HOOP_NORMALIZED, W, H, HOOP_TOKENS_SKIP_HEAD)
        print(f"[I] 从篮筐bbox读取到 {len(hoop_centers)} 个中心点")

    # ---------- 构建两套对应 ----------
    # 阶段一：只用“平面点”（不加篮筐点，保证共面）
    obj_all_s1, img_all_s1 = build_correspondences_with_hoop(
        uv_list, labs_list, names, PTS3D_REF,
        hoop_centers=None, use_hoop=False, hoop3d=HOOP3D, min_pts=6
    )
    N1 = len(obj_all_s1)
    print(f"[I] 阶段一可用帧：{N1}（仅平面点）")
    if N1 == 0:
        raise RuntimeError("阶段一没有任何一帧满足最少可见点（min_pts），无法标定。")

    # 阶段二：在平面点基础上“可选地”加上篮筐非共面点
    obj_all_s2, img_all_s2 = build_correspondences_with_hoop(
        uv_list, labs_list, names, PTS3D_REF,
        hoop_centers=hoop_centers, use_hoop=HOOP_USE, hoop3d=HOOP3D, min_pts=6
    )
    N2 = len(obj_all_s2)
    print(f"[I] 阶段二可用帧：{N2}（平面点 + {'篮筐点' if HOOP_USE else '无篮筐点'}）")
    if N2 == 0:
        raise RuntimeError("阶段二没有任何一帧满足最少可见点（min_pts），无法标定。")

    # ---------- 随机抽样 ----------
    rng = np.random.default_rng(RANDOM_SEED)

    # 阶段一的抽样
    n1 = max(2, min(N_STAGE1, N1))  # 至少两帧更稳
    idx_stage1 = rng.choice(N1, size=n1, replace=False).tolist()

    # 阶段二的抽样（是否包含阶段一）
    n2 = min(N_STAGE2, N2)
    if STAGE2_SUPERSET and N1 == N2:
        # 当两套列表一一对应时，才能确保“阶段二包含阶段一”
        # 这里两套是由相同原始序列过滤而来，顺序一致，可直接复用 idx
        if n2 <= n1:
            idx_stage2 = idx_stage1[:n2]
        else:
            remain = [i for i in range(N2) if i not in idx_stage1]
            add = rng.choice(len(remain), size=(n2 - n1), replace=False)
            idx_stage2 = idx_stage1 + [remain[j] for j in add]
    else:
        # 否则就独立在阶段二可用集合上抽样
        idx_stage2 = rng.choice(N2, size=n2, replace=False).tolist()

    # ---------- 阶段一：仅平面点标定（不需要内参初值） ----------
    obj_s1 = [obj_all_s1[i] for i in idx_stage1]
    img_s1 = [img_all_s1[i] for i in idx_stage1]
    print(f"[I] 阶段一：随机取 {len(obj_s1)} 帧（共面）做初标定...")
    rms1, K1, dist1, rvecs1, tvecs1 = calibrate(obj_s1, img_s1, (W, H), None, None, use_guess=False)
    print(f"[I] 阶段一 RMS = {rms1:.3f}px\nK=\n{K1}\ndist={dist1.ravel()}")

    # ---------- 阶段二：加入篮筐点精炼（非共面，使用上一阶段的K作为初值） ----------
    obj_s2 = [obj_all_s2[i] for i in idx_stage2]
    img_s2 = [img_all_s2[i] for i in idx_stage2]
    print(f"[I] 阶段二：随机取 {len(obj_s2)} 帧（{'含' if HOOP_USE else '不含'}篮筐点）精炼...")
    rms2, K2, dist2, rvecs2, tvecs2 = calibrate(obj_s2, img_s2, (W, H), K1, dist1, use_guess=True)
    errs2 = per_frame_error(obj_s2, img_s2, K2, dist2, rvecs2, tvecs2)
    print(f"[I] 阶段二 RMS = {rms2:.3f}px，mean={errs2.mean():.3f}, 95%={np.percentile(errs2,95):.3f}")

    # ---------- 误差剔除 → 最终精炼 ----------
    # 这里用阶段二的 per-frame 平均误差剔除“坏帧”，再精炼一次
    names_s2 = [names[i] for i in idx_stage2]  # 仅用于打印/记录
    obj_keep, img_keep, names_keep, keep_idx = prune_by_error(obj_s2, img_s2, names_s2, errs2, thr_px=ERR_THRESH_PX)
    print(f"[I] 剔除阈值 {ERR_THRESH_PX}px 后保留 {len(obj_keep)}/{len(obj_s2)} 帧")
    rmsF, KF, distF, rvecsF, tvecsF = calibrate(obj_keep, img_keep, (W, H), K2, dist2, use_guess=True)
    errsF = per_frame_error(obj_keep, img_keep, KF, distF, rvecsF, tvecsF)
    print(f"[I] 最终 RMS = {rmsF:.3f}px，mean={errsF.mean():.3f}, 95%={np.percentile(errsF,95):.3f}")
    print("[I] 最终内参 K =\n", KF)
    print("[I] 最终畸变 dist = ", distF.ravel())

    # ---------- 保存 ----------
    # print(f"[I] 正在为 {len(obj_all_s2)} 帧保存 R/t ...")
    
    # out = Path("camera_calib.yaml")
    # fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
    # fs.write("K", KF); fs.write("dist", distF)
    # fs.write("image_width", int(W)); fs.write("image_height", int(H))
    # fs.write("rms", float(rmsF))

    # # 保存所有帧外参
    # fs.startWriteStruct("frames", cv2.FileNode_SEQ)
    # for i, (rvec, tvec, name) in enumerate(zip(rvecs2, tvecs2, names)):
    #     R, _ = cv2.Rodrigues(rvec)
    #     t = tvec.reshape(3, 1)
    #     fs.startWriteStruct("", cv2.FileNode_MAP)
    #     fs.write("index", int(i))
    #     fs.write("name", name)
    #     fs.write("R", R)
    #     fs.write("t", t)
    #     fs.endWriteStruct()
    # fs.endWriteStruct()

    # fs.release()
    # print(f"[I] 已保存 {len(rvecs2)} 帧的外参到 {out.resolve()}")

        # ---------- 保存：为文件夹里所有帧求 R/t 并写入 ----------
    print("[I] 使用最终内参/畸变 (KF/distF) 对文件夹里所有帧逐帧求姿态...")

    def _solve_pose_for_frame(P3, P2, K, dist, rvec_init=None, tvec_init=None):
        """
        稳健的单帧姿态求解：
          - 平面优先 IPPE（多解取重投影误差最小）
          - 失败或无候选 → 回退到 ITERATIVE
          - 可选使用上帧初值提高稳定性
        返回: (R(3x3), t(3x1), reproj_err_px)
        """
        P3 = np.asarray(P3, np.float32).reshape(-1, 3)
        P2 = np.asarray(P2, np.float32).reshape(-1, 1, 2)
        K  = np.asarray(K,  np.float64)
        dist = np.asarray(dist, np.float64)

        if P3.shape[0] < 4:
            raise RuntimeError(f"有效点不足(>=4)，当前 {P3.shape[0]}")

        def reproj_err(rv, tv):
            proj, _ = cv2.projectPoints(P3, rv, tv, K, dist)
            return float(cv2.norm(P2, proj, cv2.NORM_L2) / len(P3))

        planar = np.max(np.abs(P3[:, 2] - np.mean(P3[:, 2]))) < 1e-7

        # 1) IPPE（平面更稳）
        if planar and hasattr(cv2, "solvePnPGeneric"):
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(P3, P2, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if ok and len(rvecs) > 0:
                cands = []
                for rv, tv in zip(rvecs, tvecs):
                    err = reproj_err(rv, tv)
                    if np.isfinite(err):
                        cands.append((err, rv, tv))
                if cands:
                    err, rv, tv = min(cands, key=lambda x: x[0])
                    R, _ = cv2.Rodrigues(rv)
                    return R, tv.reshape(3, 1), err
            # 无候选则回退

        # 2) 回退：ITERATIVE（可用初值）
        if rvec_init is not None and tvec_init is not None:
            ok, rvec, tvec = cv2.solvePnP(P3, P2, K, dist, rvec=rvec_init, tvec=tvec_init,
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(P3, P2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP 失败（IPPE/ITERATIVE 均无解）")

        err = reproj_err(rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3, 1), err

    # 逐帧构建/求解（不预过滤列表，逐帧尝试；失败不阻塞）
    all_R, all_t, all_err, all_names = [], [], [], []
    skipped = []   # 记录跳过/失败的帧及原因

    last_rvec = None
    last_tvec = None

    for uv, labs, nm in zip(uv_list, labs_list, names):
        try:
            mask = (labs == 2)
            P3 = PTS3D_REF[mask]
            P2 = uv[mask]

            # 可选加入篮筐点（提高非共面约束）
            if HOOP_USE and (hoop_centers is not None):
                stem = Path(nm).stem
                if stem in hoop_centers:
                    u_h, v_h = hoop_centers[stem]
                    P3 = np.vstack([P3, np.array(HOOP3D, dtype=np.float32)])
                    P2 = np.vstack([P2, np.array([u_h, v_h], dtype=np.float32)])

            if P3.shape[0] < 4:
                skipped.append({"name": nm, "reason": f"有效点不足({P3.shape[0]})"})
                continue

            # 形状整理
            if P2.ndim == 2:
                P2_use = P2.reshape(-1, 1, 2)
            else:
                P2_use = P2

            # 求解（用上帧作初值以提高稳健性）
            R, t, err = _solve_pose_for_frame(P3, P2_use, KF, distF, rvec_init=last_rvec, tvec_init=last_tvec)

            # 更新下一帧初值
            last_rvec, _ = cv2.Rodrigues(R)
            last_tvec = t.copy()

            all_R.append(R); all_t.append(t); all_err.append(err); all_names.append(nm)

        except Exception as e:
            skipped.append({"name": nm, "reason": str(e)})

    print(f"[I] 成功 {len(all_R)} 帧，跳过 {len(skipped)} 帧；平均误差 = {np.mean(all_err) if all_err else float('nan'):.3f}px")

    # 写入 YAML
    out = Path("camera_calib.yaml")
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)

    # 全局参数
    fs.write("K", KF)
    fs.write("dist", distF)
    fs.write("image_width", int(W))
    fs.write("image_height", int(H))
    fs.write("rms", float(rmsF))

    # 成功帧
    fs.startWriteStruct("frames_all", cv2.FileNode_SEQ)
    for i, (nm, R, t, err) in enumerate(zip(all_names, all_R, all_t, all_err)):
        fs.startWriteStruct("", cv2.FileNode_MAP)
        fs.write("index", int(i))
        fs.write("name", nm)
        fs.write("R", R)
        fs.write("t", t)
        fs.write("reproj_err_px", float(err))
        fs.endWriteStruct()
    fs.endWriteStruct()

    # 跳过帧（可选，便于排查）
    fs.startWriteStruct("skipped_frames", cv2.FileNode_SEQ)
    for item in skipped:
        fs.startWriteStruct("", cv2.FileNode_MAP)
        fs.write("name", item["name"])
        fs.write("reason", item["reason"])
        fs.endWriteStruct()
    fs.endWriteStruct()

    fs.release()
    print(f"[I] 已写入 {out.resolve()}：frames_all={len(all_R)}, skipped={len(skipped)}")



if __name__ == "__main__":
    main()
