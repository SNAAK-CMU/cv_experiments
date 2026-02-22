"""Calculate per-file and aggregate IOU for a specific class color between
predicted mask images and ground-truth mask images.

Usage (defaults can be overridden via CLI):
  python calc_IOU.py --pred_dir /path/to/pred --gt_dir /path/to/gt --color 255,0,0

Globals you can edit in the file directly:
  PRED_MASKS_DIR: default directory containing predicted masks
  GT_MASKS_DIR: default directory containing ground-truth masks
  TARGET_COLOR: BGR tuple for the class color to evaluate (e.g. (255,0,0))

The script matches files by basename. If image sizes differ the predicted mask
will be resized (nearest neighbor) to the GT size before IOU computation.
If both predicted and GT masks have zero union (both empty) IoU is treated as 1.0.
"""

import os
import csv
import argparse
from typing import Tuple, List

import numpy as np
import cv2


# Default globals (edit if you prefer hard-coded defaults)
PRED_MASKS_DIR = "/home/snaak/Documents/Abhi/Vision/results/pred_masks"
GT_MASKS_DIR = "/home/snaak/Documents/Abhi/Vision/data/gt_masks"
# TARGET_COLOR is in BGR (OpenCV) order. Example: pure red in BGR is (0,0,255)
TARGET_COLOR = (0, 0, 255)


def parse_color(s: str) -> Tuple[int, int, int]:
	"""Parse a color string into a BGR tuple. Accepts "R,G,B" or "#RRGGBB".
	Returns BGR (int,int,int) for OpenCV usage.
	"""
	s = s.strip()
	if s.startswith("#"):
		# hex is RRGGBB
		hexv = s[1:]
		if len(hexv) != 6:
			raise argparse.ArgumentTypeError("Hex color must be 6 digits, e.g. #ff0000")
		r = int(hexv[0:2], 16)
		g = int(hexv[2:4], 16)
		b = int(hexv[4:6], 16)
		return (b, g, r)
	parts = s.split(",")
	if len(parts) != 3:
		raise argparse.ArgumentTypeError("Color must be R,G,B or #RRGGBB")
	r, g, b = [int(p) for p in parts]
	return (b, g, r)


def load_image(path: str):
	# Load with unchanged flag to preserve single-channel masks if present
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	return img


def extract_class_mask(img: np.ndarray, target_color: Tuple[int, int, int]) -> np.ndarray:
	"""Return a boolean mask (dtype=bool) where pixels match target_color.
	Handles single-channel, 3-channel, or 4-channel images.
	"""
	if img is None:
		raise ValueError("img is None")
	if img.ndim == 2:
		# grayscale: treat non-zero pixels as class
		return img.astype(bool)
	# If image has alpha channel, ignore alpha
	if img.shape[2] >= 3:
		rgb = img[:, :, :3]
		# img is in BGR, target_color is BGR
		t = np.array(target_color, dtype=rgb.dtype).reshape((1, 1, 3))
		match = np.all(rgb == t, axis=2)
		return match
	# Fallback: treat non-zero
	return np.any(img != 0, axis=2)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
	"""Compute IoU between two boolean masks."""
	pred_bool = pred_mask.astype(bool)
	gt_bool = gt_mask.astype(bool)
	inter = np.logical_and(pred_bool, gt_bool).sum()
	union = np.logical_or(pred_bool, gt_bool).sum()
	if union == 0:
		# Both empty: define IoU as 1.0 (perfect) else 0.0 if only one empty
		return 1.0 if inter == 0 else 0.0
	return float(inter) / float(union)


def find_matching_files(pred_dir: str, gt_dir: str, exts: Tuple[str, ...], recursive: bool) -> List[Tuple[str, str]]:
	"""Return list of (pred_path, gt_path) pairs matching basenames present in both dirs.
	If recursive, walks pred_dir recursively and computes relative path to match in gt_dir.
	"""
	pairs = []
	if recursive:
		for root, _, files in os.walk(pred_dir):
			for f in sorted(files):
				if not f.lower().endswith(exts):
					continue
				rel = os.path.relpath(os.path.join(root, f), pred_dir)
				pred_path = os.path.join(root, f)
				gt_path = os.path.join(gt_dir, rel)
				if os.path.exists(gt_path):
					pairs.append((pred_path, gt_path))
				else:
					# try matching by basename only
					candidate = os.path.join(gt_dir, os.path.basename(f))
					if os.path.exists(candidate):
						pairs.append((pred_path, candidate))
	else:
		pred_files = [f for f in sorted(os.listdir(pred_dir)) if f.lower().endswith(exts)]
		for f in pred_files:
			pred_path = os.path.join(pred_dir, f)
			gt_path = os.path.join(gt_dir, f)
			if os.path.exists(gt_path):
				pairs.append((pred_path, gt_path))
	return pairs


def main(pred_dir: str, gt_dir: str, color: Tuple[int, int, int], out_csv: str = None, recursive: bool = False):
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
	pairs = find_matching_files(pred_dir, gt_dir, exts, recursive)
	if not pairs:
		print("No matching files found between predicted and GT dirs.")
		return

	results = []
	for pred_path, gt_path in pairs:
		pred_img = load_image(pred_path)
		gt_img = load_image(gt_path)
		if gt_img is None:
			print(f"Warning: failed to read GT image {gt_path}, skipping")
			continue
		if pred_img is None:
			print(f"Warning: failed to read predicted image {pred_path}, skipping")
			continue

		# If sizes differ, resize predicted to GT size using nearest neighbor
		if pred_img.shape[0:2] != gt_img.shape[0:2]:
			pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

		pred_mask = extract_class_mask(pred_img, color)
		gt_mask = extract_class_mask(gt_img, color)
		iou = compute_iou(pred_mask, gt_mask)
		results.append((os.path.basename(pred_path), iou))
		print(f"{os.path.basename(pred_path)} -> IoU: {iou:.4f}")

	ious = [r[1] for r in results]
	if ious:
		mean_iou = float(np.mean(ious))
		median_iou = float(np.median(ious))
		print(f"Processed {len(ious)} files. Mean IoU: {mean_iou:.4f}, Median IoU: {median_iou:.4f}")
	else:
		print("No IoU scores computed")

	if out_csv and results:
		with open(out_csv, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['file', 'iou'])
			writer.writerows(results)
		print(f"Wrote per-file IoU to {out_csv}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculate IoU for color-coded masks')
	parser.add_argument('--pred_dir', type=str, default=PRED_MASKS_DIR, help='Predicted masks directory')
	parser.add_argument('--gt_dir', type=str, default=GT_MASKS_DIR, help='Ground-truth masks directory')
	parser.add_argument('--color', type=parse_color, default=TARGET_COLOR, help='Target class color as R,G,B or #RRGGBB')
	parser.add_argument('--out_csv', type=str, default=None, help='Optional CSV path to save per-file IoU')
	parser.add_argument('--recursive', action='store_true', help='Recursively walk predicted directory and mirror structure when matching')
	args = parser.parse_args()
	main(args.pred_dir, args.gt_dir, args.color, args.out_csv, args.recursive)

