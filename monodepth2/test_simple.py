# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Licensed under the Monodepth2 licence (non-commercial use only)

from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Monodepth2 models.')

    parser.add_argument('--image_path', type=str, required=True,
                        help='path to a test image OR a folder of images')

    # 方式一：官方预训练（只给 model_name）
    parser.add_argument('--model_name', type=str, default=None,
                        help='pretrained model name (if set, will download/use Niantic weights). '
                             'Ignored when --load_weights_folder is provided.')

    # 方式二：自训练权重（优先级更高）
    parser.add_argument('--load_weights_folder', type=str, default=None,
                        help='path to your trained weights folder, e.g. /path/to/weights_19')

    parser.add_argument('--num_layers', type=int, default=18,
                        help='resnet layers for your custom model (18 or 50)')

    parser.add_argument('--ext', type=str, default=None,
                        help='image extension to search for when image_path is a folder '
                             '(default: scan jpg/jpeg/png)')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='where to save outputs (default: alongside inputs)')

    parser.add_argument('--no_cuda', action='store_true',
                        help='if set, disables CUDA')

    parser.add_argument('--pred_metric_depth', action='store_true',
                        help='predict metric depth instead of disparity '
                             '(only makes sense for stereo-trained models)')

    return parser.parse_args()


def load_custom_model(device, weights_folder, num_layers):
    """Load a model from a custom weights folder (encoder.pth + depth.pth)."""
    print("-> Loading model from custom folder:", weights_folder)
    enc_path = os.path.join(weights_folder, "encoder.pth")
    dec_path = os.path.join(weights_folder, "depth.pth")

    if not (os.path.isfile(enc_path) and os.path.isfile(dec_path)):
        raise FileNotFoundError(
            f"encoder/depth weights not found under: {weights_folder}")

    enc_state = torch.load(enc_path, map_location=device)

    # image size used in training (saved by trainer for encoder)
    feed_height = int(enc_state.get('height', 192))
    feed_width = int(enc_state.get('width', 640))

    encoder = networks.ResnetEncoder(num_layers, False)
    # filter keys (state dict may include meta like height/width)
    enc_state_filtered = {k: v for k, v in enc_state.items()
                          if k in encoder.state_dict()}
    encoder.load_state_dict({**encoder.state_dict(), **enc_state_filtered})
    encoder.to(device).eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    depth_decoder.load_state_dict(torch.load(dec_path, map_location=device))
    depth_decoder.to(device).eval()

    return encoder, depth_decoder, feed_height, feed_width


def load_pretrained_model(device, model_name):
    """Load one of Niantic's pretrained models by name."""
    assert model_name is not None, "model_name must be provided for pretrained mode"
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading pretrained model:", model_path)

    enc_path = os.path.join(model_path, "encoder.pth")
    dec_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    enc_state = torch.load(enc_path, map_location=device)
    feed_height = int(enc_state['height'])
    feed_width = int(enc_state['width'])

    enc_state_filtered = {k: v for k, v in enc_state.items()
                          if k in encoder.state_dict()}
    encoder.load_state_dict(enc_state_filtered)
    encoder.to(device).eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    depth_decoder.load_state_dict(torch.load(dec_path, map_location=device))
    depth_decoder.to(device).eval()

    return encoder, depth_decoder, feed_height, feed_width


def collect_image_paths(image_path, ext):
    """Return a list of image file paths and an output directory."""
    if os.path.isfile(image_path):
        return [image_path], os.path.dirname(image_path)

    if os.path.isdir(image_path):
        if ext is None:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            paths = []
            for e in exts:
                paths += glob.glob(os.path.join(image_path, e))
        else:
            paths = glob.glob(os.path.join(image_path, f"*.{ext}"))
        paths = sorted(paths)
        return paths, image_path

    raise FileNotFoundError(f"image_path not found: {image_path}")


def save_outputs(disp, disp_resized, output_directory, output_name, pred_metric_depth):
    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

    if pred_metric_depth:
        npy_path = os.path.join(output_directory, f"{output_name}_depth.npy")
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        np.save(npy_path, metric_depth)
    else:
        npy_path = os.path.join(output_directory, f"{output_name}_disp.npy")
        np.save(npy_path, scaled_disp.cpu().numpy())

    # color map preview
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    jpg_path = os.path.join(output_directory, f"{output_name}_disp.jpeg")
    im.save(jpg_path)

    return jpg_path, npy_path


def test_simple(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # load model
    if args.load_weights_folder is not None:
        encoder, depth_decoder, feed_h, feed_w = load_custom_model(
            device, args.load_weights_folder, args.num_layers)
    else:
        encoder, depth_decoder, feed_h, feed_w = load_pretrained_model(
            device, args.model_name)

    # gather inputs
    paths, default_out_dir = collect_image_paths(args.image_path, args.ext)
    out_dir = args.output_dir or default_out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"-> Predicting on {len(paths)} images")
    print(f"-> Outputs will be saved to: {out_dir}")

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for i, img_path in enumerate(paths):
            if img_path.endswith("_disp.jpg") or img_path.endswith("_disp.jpeg"):
                continue

            # load & resize
            img = pil.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_resized = img.resize((feed_w, feed_h), pil.LANCZOS)
            inp = to_tensor(img_resized).unsqueeze(0).to(device)

            # forward
            feats = encoder(inp)
            outputs = depth_decoder(feats)
            disp = outputs[("disp", 0)]

            # resize back to original for visualization
            disp_resized = torch.nn.functional.interpolate(
                disp, (orig_h, orig_w), mode="bilinear", align_corners=False)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            jpg_path, npy_path = save_outputs(
                disp, disp_resized, out_dir, stem, args.pred_metric_depth)

            print(f"   [{i+1}/{len(paths)}] saved:")
            print(f"     - {jpg_path}")
            print(f"     - {npy_path}")

    print("-> Done!")


if __name__ == "__main__":
    args = parse_args()
    test_simple(args)
