from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from cv2 import imread
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).float()
    if map.max() == map.min():
        idx = torch.zeros_like(map, dtype=torch.long)
    else:
        map_n = (map - map.min()) / (map.max() - map.min() + 1e-6)
        idx = (map_n * (colors.shape[0] - 1)).round().long().squeeze()
    img = colors[idx]
    return img


def visual_masks(masks):
    # masks: torch tensor bool of shape (k,h,w) or (h,w)
    if masks.ndim == 2:
        h, w = masks.shape
        total_mask = masks.long()
    else:
        h, w = masks.shape[-2:]
        k = masks.shape[0]
        total_mask = torch.zeros((h, w), dtype=torch.long)
        for i, m in enumerate(masks):
            total_mask = total_mask + m.long() * (((i + k) * i) + 1)
    alpha = (total_mask != 0)
    img = colormap(total_mask)
    img = torch.cat((img, alpha.unsqueeze(-1).float()), dim=-1)
    return img


def glob_data(data_dir):
    data_paths = sorted(glob(data_dir))
    return data_paths


def main():
    parser = ArgumentParser(description="Visualize SAM masks for one image")
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True, help='Path to the images folder.')
    parser.add_argument('--index', type=int, default=0, help='Index of image in folder to process')
    args = parser.parse_args()

    print("Initializing SAM...")
    sam = build_sam(checkpoint=args.sam_checkpoint).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=.9)

    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    image_paths = [p for p in glob_data(os.path.join(args.file_path, "*.*"))
                   if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
    if len(image_paths) == 0:
        print("No images found in", args.file_path)
        return
    idx = min(args.index, len(image_paths) - 1)
    image_path = image_paths[idx]
    image_name = os.path.basename(image_path).split('.')[0]

    print(f"Processing image: {image_path}")
    image = imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    results = mask_generator.generate(image)

    # build masks array same as get_sam_masks.py
    masks = []
    for r in results:
        masks.append(r['segmentation'].astype(bool))
    masks = np.array(masks)
    if masks.size == 0:
        print("No masks produced for image")
        return

    out_base = os.path.join('output', 'visualize_sam', image_name)
    orig_dir = os.path.join(out_base, 'original')
    merged_dir = os.path.join(out_base, 'merged')
    unions_dir = os.path.join(out_base, 'unions')
    txt_dir = os.path.join(out_base, 'txt')
    for d in (orig_dir, merged_dir, unions_dir, txt_dir):
        os.makedirs(d, exist_ok=True)

    # Save original masks
    for i, m in enumerate(masks):
        np.save(os.path.join(orig_dir, f'mask_{i}.npy'), m)
        np.savetxt(os.path.join(txt_dir, f'orig_mask_{i}.txt'), m.astype(int), fmt='%d')
        plt.imsave(os.path.join(orig_dir, f'mask_{i}.png'), m.astype(float), cmap='gray')

    # Convert to boolean numpy for merging ops
    merged = masks.copy()
    K = merged.shape[0]
    for m_idx in range(K):
        mask = merged[m_idx]
        mask_sum = mask.sum()
        if m_idx + 1 >= K:
            continue
        # compute overlaps with subsequent masks
        subsequent = merged[m_idx + 1:]
        overlaps = (mask & subsequent).sum(axis=(1, 2))
        np.savetxt(os.path.join(txt_dir, f'overlaps_{m_idx}.txt'), overlaps.astype(int), fmt='%d')
        for j, ov in enumerate(overlaps):
            j_idx = m_idx + 1 + j
            union_bool = mask & merged[j_idx]
            if ov > 0:
                plt.imsave(os.path.join(unions_dir, f'union_{m_idx}_{j_idx}.png'), union_bool.astype(float), cmap='gray')
                np.savetxt(os.path.join(txt_dir, f'union_{m_idx}_{j_idx}.txt'), union_bool.astype(int), fmt='%d')
            # apply same merge rule: if overlap > 0.9 * mask.sum -> OR into subsequent
            if mask_sum > 0 and ov > 0.9 * mask_sum:
                merged[j_idx] = merged[j_idx] | mask

    # Save merged masks
    for i, m in enumerate(merged):
        np.save(os.path.join(merged_dir, f'mask_{i}.npy'), m)
        np.savetxt(os.path.join(txt_dir, f'merged_mask_{i}.txt'), m.astype(int), fmt='%d')
        plt.imsave(os.path.join(merged_dir, f'mask_{i}.png'), m.astype(float), cmap='gray')

    # Save composite visualizations (before and after)
    try:
        import torch as _torch
        orig_t = _torch.from_numpy(masks).cuda() if _torch.cuda.is_available() else _torch.from_numpy(masks)
        merged_t = _torch.from_numpy(merged).cuda() if _torch.cuda.is_available() else _torch.from_numpy(merged)
        vis_orig = visual_masks(orig_t.cpu())
        vis_merged = visual_masks(merged_t.cpu())
        plt.imsave(os.path.join(out_base, 'composite_original.png'), vis_orig.cpu().numpy())
        plt.imsave(os.path.join(out_base, 'composite_merged.png'), vis_merged.cpu().numpy())
    except Exception:
        # fallback simple composite: index map
        H, W = masks.shape[1:]
        total_orig = np.zeros((H, W), dtype=np.int32)
        total_merged = np.zeros((H, W), dtype=np.int32)
        for i in range(K):
            total_orig += masks[i].astype(int) * (i + 1)
            total_merged += merged[i].astype(int) * (i + 1)
        plt.imsave(os.path.join(out_base, 'composite_original.png'), total_orig, cmap='turbo')
        plt.imsave(os.path.join(out_base, 'composite_merged.png'), total_merged, cmap='turbo')

    print(f"Saved visualization outputs to {out_base}")


if __name__ == '__main__':
    main()
