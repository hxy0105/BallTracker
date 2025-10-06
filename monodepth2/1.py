from test_simple import parse_args, test_simple

class Args:
    pass

args = Args()
args.image_path = r"E:\balltracker\ball_tracker\images\dataset_1000_to_tag"
args.load_weights_folder = r"E:\balltracker\ball_tracker\monodepth2_logs\monodepth2_logs\my_mono_model\models\weights_19"
args.num_layers = 18
args.output_dir = r"E:\balltracker\monodepth2_outputs"
args.no_cuda = False
args.pred_metric_depth = False
args.ext = "png"  # 你的图片扩展名（如果是jpg改成jpg）

test_simple(args)

run_test_from_split.py

E:\balltracker\ball_tracker\images\dataset_1000_to_tag\2023_abu-dhabi_Chonchi_vs_Abu_Dhabi_Qualifying_Draw_A.mp4_frame_6083.png

# import os
# import torch
# from PIL import Image as pil
# from torchvision import transforms

# # 这三个函数来自“我给你的改好的 test_simple.py”
# from test_simple import load_custom_model, load_pretrained_model, save_outputs

# def run_from_split(
#     split_file,
#     base_dir,                       # test_files.txt 里的相对路径相对于这个 base_dir
#     output_dir,
#     weights_folder=None,            # 自训练模型权重：.../models/weights_XX
#     model_name=None,                # 或者官方预训练名：mono_640x192 等（二选一）
#     num_layers=18,
#     no_cuda=False,
#     pred_metric_depth=False,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

#     # 1) 加载模型（只加载一次）
#     if weights_folder is not None:
#         encoder, depth_decoder, feed_h, feed_w = load_custom_model(device, weights_folder, num_layers)
#     else:
#         encoder, depth_decoder, feed_h, feed_w = load_pretrained_model(device, model_name)

#     # 2) 读取 split 列表并拼成绝对路径
#     with open(split_file, "r", encoding="utf-8") as f:
#         rel_paths = [ln.strip() for ln in f if ln.strip()]

#     img_paths = [
#         os.path.join(base_dir, p.replace("\\", "/"))
#         for p in rel_paths
#     ]

#     os.makedirs(output_dir, exist_ok=True)
#     to_tensor = transforms.ToTensor()

#     ok, failed = 0, []

#     # 3) 逐张推理
#     for i, img_path in enumerate(img_paths):
#         try:
#             if img_path.endswith("_disp.jpg") or img_path.endswith("_disp.jpeg"):
#                 continue

#             img = pil.open(img_path).convert("RGB")
#             ow, oh = img.size
#             img_resized = img.resize((feed_w, feed_h), pil.LANCZOS)
#             inp = to_tensor(img_resized).unsqueeze(0).to(device)

#             feats = encoder(inp)
#             outputs = depth_decoder(feats)
#             disp = outputs[("disp", 0)]

#             disp_resized = torch.nn.functional.interpolate(
#                 disp, (oh, ow), mode="bilinear", align_corners=False
#             )

#             stem = os.path.splitext(os.path.basename(img_path))[0]
#             save_outputs(disp, disp_resized, output_dir, stem, pred_metric_depth)
#             ok += 1

#             if (i + 1) % 20 == 0:
#                 print(f"[{i+1}/{len(img_paths)}] done")

#         except Exception as e:
#             failed.append((img_path, str(e)))

#     print(f"\n[SUMMARY] success: {ok}, failed: {len(failed)}")
#     if failed:
#         print("First few failures:")
#         for p, err in failed[:5]:
#             print(" -", p, "|", err)
#     print("Outputs:", output_dir)


# if __name__ == "__main__":
#     # ==== 按你的实际路径修改这四个 ====
#     SPLIT_FILE  = r"E:\balltracker\ball_tracker\monodepth2\monodepth2\splits\custom\test_files.txt"
#     BASE_DIR    = r"E:\balltracker\ball_tracker"     # test_files.txt 里的相对路径是相对它的
#     OUTPUT_DIR  = r"E:\balltracker\monodepth2\monodepth2_outputs"

#     WEIGHTS     = r"E:\balltracker\ball_tracker\monodepth2_logs\monodepth2_logs\my_mono_model\models\weights_19"
#     NUM_LAYERS  = 18          # 如果训练用的是 ResNet50，这里改成 50

#     run_from_split(
#         split_file=SPLIT_FILE,
#         base_dir=BASE_DIR,
#         output_dir=OUTPUT_DIR,
#         weights_folder=WEIGHTS,    # 用自训练模型
#         model_name=None,           # 若想用官方预训练，就把上面 weights_folder=None，改成 model_name="mono_640x192"
#         num_layers=NUM_LAYERS,
#         no_cuda=False,
#         pred_metric_depth=False,
#     )
