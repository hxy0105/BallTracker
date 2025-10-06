import cv2
import numpy as np
from pathlib import Path

# ========== 配置 ==========
YAML_PATH   = r"E:\balltracker\ball_tracker\camera_calib.yaml"
TARGET_NAME = r"2023_abu-dhabi_Jeddah_vs_Chonchi_Qualifying_Draw_A.mp4_frame_18334.txt"
RIM_FOLDER  = r"E:\balltracker\ball_tracker\labels\data\3x3\labels\final_extraction\all_annotations\rim"
BALL_FOLDER = r"E:\balltracker\ball_tracker\labels\data\3x3\labels\final_extraction\all_annotations\sportsball"
W, H        = 1920, 1080
NORM_2D     = True
RIM_WORLD   = (7.50, 1.575, 3.05)
DEPTH_PATH  = r"E:\balltracker\ball_tracker\images\dataset_1000_to_tag\2023_abu-dhabi_Jeddah_vs_Chonchi_Qualifying_Draw_A.mp4_frame_18334_disp.jpeg"
INVERSE_DEPTH = True  # 若是逆深度(1/z)，改 True

# ========== 工具 ==========
def load_depth(path, target_size=None):
    """读取深度/视差图并转换为单通道 float32，并可自动调整尺寸。"""
    if str(path).lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"读深度失败: {path}")
        img = img.astype(np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        arr = img

    # 归一化
    mx = float(arr.max())
    if mx > 0:
        arr /= mx

    # ✅ resize 到目标尺寸（如1920×1080）
    if target_size is not None:
        arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR)

    return arr


def bilinear_sample(depth, u, v):
    """双线性采样，保证返回python float。"""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return float("nan")
    x0 = int(np.floor(u)); x1 = min(x0+1, w-1)
    y0 = int(np.floor(v)); y1 = min(y0+1, h-1)
    wa = (x1-u)*(y1-v); wb = (u-x0)*(y1-v); wc = (x1-u)*(v-y0); wd = (u-x0)*(v-y0)
    val = wa*depth[y0, x0] + wb*depth[y0, x1] + wc*depth[y1, x0] + wd*depth[y1, x1]
    return float(np.asarray(val).mean())  # 永远返回标量

def undistort_to_normalized(uv, K, dist):
    uv = np.ascontiguousarray(uv.reshape(-1,1,2).astype(np.float32))
    pts = cv2.undistortPoints(uv, K, dist)  # -> (N,1,2) 归一化坐标
    return pts.reshape(-1,2)

def world_to_cam(R, t, Xw):
    return R @ Xw + t

def cam_to_world(R, t, Xc):
    return R.T @ (Xc - t)

def read_yolo_center(txt_path, W, H, normalized=True, skip_head=0):
    """返回 (u,v) 像素中心。默认取第一行标注。YOLO: cls cx cy w h"""
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到标注: {txt_path}")
    line = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not line:
        raise ValueError(f"空文件: {txt_path}")
    toks = [t for t in line[0].replace(",", " ").split() if t]
    if skip_head > 0:
        toks = toks[skip_head:]
    if len(toks) < 5:
        raise ValueError(f"格式不对(需>=5个数字): {txt_path}")
    cx, cy = float(toks[1]), float(toks[2])
    if normalized:
        u = cx * W
        v = cy * H
    else:
        u = cx; v = cy
    return float(u), float(v)

def find_frame_in_yaml(yaml_path, target_name):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开YAML: {yaml_path}")
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    node = fs.getNode("frames")
    if node.empty():
        node = fs.getNode("frames_all")
    if node.empty():
        fs.release()
        raise KeyError("YAML中未找到 frames / frames_all 节点")
    N = int(node.size())
    for i in range(N):
        fi = node.at(i)
        name = fi.getNode("name").string()
        if name == target_name:
            R = fi.getNode("R").mat()
            t = fi.getNode("t").mat()
            fs.release()
            return K.astype(np.float64), dist.astype(np.float64), R.astype(np.float64), t.astype(np.float64)
    fs.release()
    raise KeyError(f"未在 YAML 中找到 name == '{target_name}' 的帧")

def read_yolo_top_center(txt_path, W, H, normalized=True, skip_head=0, clip=True):
    """
    返回 bbox 上沿中心 (u,v)。YOLO: cls cx cy w h
    若 normalized=True，(cx,cy,w,h) 是 0~1；否则是像素。
    clip=True 时会把 (u,v) 裁剪到画幅内，避免越界取样。
    """
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到标注: {txt_path}")
    line = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not line:
        raise ValueError(f"空文件: {txt_path}")
    toks = [t for t in line[0].replace(",", " ").split() if t]
    if skip_head > 0:
        toks = toks[skip_head:]
    if len(toks) < 5:
        raise ValueError(f"格式不对(需>=5个数字): {txt_path}")

    cx, cy, bw, bh = map(float, toks[1:5])
    if normalized:
        u = cx * W
        v = (cy - bh/2.0) * H     # ← 上沿中心
    else:
        u = cx
        v = cy - bh/2.0

    if clip:
        u = float(np.clip(u, 0, W-1))
        v = float(np.clip(v, 0, H-1))
    return u, v


# ========== 主流程 ==========
def main():
    # 1) 读取该帧 K, dist, R, t
    K, dist, R, t = find_frame_in_yaml(YAML_PATH, TARGET_NAME)
    t = t.reshape(3,)

    # 2) 读取该帧的篮筐/篮球像素中心
    stem = Path(TARGET_NAME).stem
    rim_txt  = Path(RIM_FOLDER)  / f"{stem}.txt"
    ball_txt = Path(BALL_FOLDER) / f"{stem}.txt"
    #u_r, v_r = read_yolo_top_center(rim_txt, W, H, normalized=NORM_2D)
    u_r, v_r = read_yolo_center(rim_txt,  W, H, normalized=NORM_2D)
    u_b, v_b = read_yolo_center(ball_txt, W, H, normalized=NORM_2D)

    # 3) 读取深度并采样（保证标量）
    depth = load_depth(DEPTH_PATH, target_size=(W, H))

    D_rim  = bilinear_sample(depth, u_r, v_r)
    D_ball = bilinear_sample(depth, u_b, v_b)

    # 判断有效性（现在 D_* 一定是 float）
    if not (np.isfinite(D_rim) and np.isfinite(D_ball)):
        raise ValueError("深度采样失败(越界/NaN)，请检查分辨率和坐标是否对齐。")

    if INVERSE_DEPTH:
        if abs(D_rim) < 1e-12 or abs(D_ball) < 1e-12:
            raise ValueError("逆深度为0，无法取倒数。")
        D_rim  = 1.0 / D_rim
        D_ball = 1.0 / D_ball

    # 4) 世界→相机：篮筐深度
    Xw_rim = np.array(RIM_WORLD, dtype=np.float64)
    Xc_rim = world_to_cam(R, t, Xw_rim)
    Z_rim  = float(Xc_rim[2])
    if Z_rim <= 0:
        raise ValueError(f"Z_rim={Z_rim:.6f} 非正，检查外参或坐标系。")

    # 5) 用篮筐做尺度标定
    if abs(D_rim) < 1e-12:
        raise ValueError("D_rim≈0，无法计算尺度。")
    alpha  = Z_rim / D_rim
    Z_ball = alpha * D_ball

    # 6) 像素→归一化射线→相机坐标
    xn, yn = undistort_to_normalized(np.array([[u_b, v_b]], dtype=np.float32), K, dist)[0]
    Xc_ball = np.array([xn * Z_ball, yn * Z_ball, Z_ball], dtype=np.float64)

    # 7) 相机→世界
    Xw_ball = cam_to_world(R, t, Xc_ball)
    Xw_ball[2] *= -1  # 让世界Z朝上

    # 打印结果
    print("\n=== 结果 ===")
    print(f"帧名: {TARGET_NAME}")
    print(f"篮筐像素(u,v) = ({u_r:.2f}, {v_r:.2f}), 篮筐世界 = {RIM_WORLD}, Z_rim = {Z_rim:.4f} m")
    print(f"篮球像素(u,v) = ({u_b:.2f}, {v_b:.2f})")
    print(f"D_rim = {D_rim:.6f}, D_ball = {D_ball:.6f}, alpha = {alpha:.6f}, Z_ball = {Z_ball:.6f} m")
    print("球的相机坐标 Xc_ball [m] =", Xc_ball)
    print("球的世界坐标 Xw_ball [m] =", Xw_ball)

        # 8) 世界坐标 -> 像素坐标（验证投影）
    Xw_ball_reshaped = Xw_ball.reshape(1, 1, 3)
    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(Xw_ball_reshaped, rvec, t, K, dist)
    u_proj, v_proj = img_pts[0, 0]

    print(f"反投影后的像素坐标 (u,v) = ({u_proj:.2f}, {v_proj:.2f})")

    def compute_2d_error(pt1, pt2):
    #"""计算两点间像素误差（欧式距离）"""
        pt1 = np.array(pt1, dtype=np.float64)
        pt2 = np.array(pt2, dtype=np.float64)
        return float(np.linalg.norm(pt1 - pt2))


    # ========== 示例调用（放在 main() 最后） ==========
    # 你原本的计算结果

    err = compute_2d_error((u_proj, v_proj), (u_b, v_b))
    print(f"预测点: ({u_proj:.2f}, {v_proj:.2f})")
    print(f"真实点: ({u_b:.2f}, {v_b:.2f})")
    print(f"像素误差 = {err:.2f} px")



if __name__ == "__main__":
    main()
