"""Calculate Euclidean distance between pickup points computed from predicted
and ground-truth binary masks.

Pickup point computation follows the same heuristic as the vision node / UNet
code: take the largest contour (or non-zero pixels) in the binary mask and
compute the centroid by averaging x and y coordinates of mask pixels.

The script matches files by basename (optionally recursively) and writes a
CSV with per-file distances plus a summary (mean/median). Optionally converts
pixel distances to metres if a pixels_to_m scale is provided.
"""

import os
import argparse
import csv
import math
from typing import List, Tuple, Optional

import numpy as np
import cv2


# Default paths (edit as needed)
PRED_MASKS_DIR = "/home/snaak/Documents/datasets/testsets/SCH_images_041525_1_T/multi_cheese_model/pred_masks/"
GT_MASKS_DIR = "/home/snaak/Documents/datasets/testsets/SCH_images_041525_1_T/multi_cheese_model/pred_binary_masks/"


def load_mask(path: str) -> Optional[np.ndarray]:
	"""Load an image and return a binary mask (dtype=uint8) where foreground
	pixels are 255 and background 0. Returns None if the image can't be read.
	"""
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		return None
	# If multi-channel, convert to grayscale by checking any nonzero in channels
	if img.ndim == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	# Threshold: treat non-zero as foreground
	_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
	return binary


def pickup_point_from_mask(binary_mask: np.ndarray) -> Optional[Tuple[float, float]]:
	"""Compute pickup point (x,y) in pixel coordinates from a binary mask.
	Strategy:
	  - find external contours
	  - if none, but mask has non-zero pixels, compute mean of pixel coordinates
	  - otherwise, take largest contour, compute centroid as mean of pixels inside it
	Returns (x, y) or None if mask empty.
	"""
	if binary_mask is None:
		return None
	# Ensure binary values are 0 or 255
	mask = (binary_mask > 0).astype(np.uint8) * 255
	if mask.sum() == 0:
		return None

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largest = max(contours, key=cv2.contourArea)
		# Create empty mask for largest contour and compute its pixel coordinates
		cmask = np.zeros_like(mask)
		cv2.drawContours(cmask, [largest], -1, 255, thickness=cv2.FILLED)
		ys, xs = np.where(cmask == 255)
		if xs.size == 0:
			return None
		cx = float(np.mean(xs))
		cy = float(np.mean(ys))
		return cx, cy
	else:
		# fallback: average of all foreground pixels
		ys, xs = np.where(mask == 255)
		if xs.size == 0:
			return None
		cx = float(np.mean(xs))
		cy = float(np.mean(ys))
		return cx, cy


def find_matching_files(pred_dir: str, gt_dir: str, exts: Tuple[str, ...], recursive: bool) -> List[Tuple[str, str]]:
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
					candidate = os.path.join(gt_dir, os.path.basename(f))
					if os.path.exists(candidate):
						pairs.append((pred_path, candidate))
	else:
		for f in sorted(os.listdir(pred_dir)):
			if not f.lower().endswith(exts):
				continue
			pred_path = os.path.join(pred_dir, f)
			gt_path = os.path.join(gt_dir, f)
			if os.path.exists(gt_path):
				pairs.append((pred_path, gt_path))
	return pairs


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


def main(pred_dir: str, gt_dir: str, out_csv: Optional[str], pixels_to_m: Optional[float], recursive: bool):
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
	pairs = find_matching_files(pred_dir, gt_dir, exts, recursive)
	if not pairs:
		print("No matching files found")
		return

	rows = []
	distances = []
	for pred_path, gt_path in pairs:
		basename = os.path.basename(pred_path)
		pred_mask = load_mask(pred_path)
		gt_mask = load_mask(gt_path)
		if pred_mask is None:
			print(f"Warning: failed to read predicted mask {pred_path}; skipping")
			continue
		if gt_mask is None:
			print(f"Warning: failed to read GT mask {gt_path}; skipping")
			continue

		pred_pt = pickup_point_from_mask(pred_mask)
		gt_pt = pickup_point_from_mask(gt_mask)

		if pred_pt is None or gt_pt is None:
			print(f"Warning: missing pickup point for {basename} (pred: {pred_pt}, gt: {gt_pt})")
			rows.append((basename, None, None, None, None))
			continue

		pix_dist = euclidean(pred_pt, gt_pt)
		m_dist = pix_dist * pixels_to_m if pixels_to_m is not None else None
		distances.append(pix_dist)
		rows.append((basename, pred_pt[0], pred_pt[1], gt_pt[0], gt_pt[1], pix_dist, m_dist))
		print(f"{basename}: pixel_distance={pix_dist:.2f}" + (f", metres={m_dist:.4f}" if m_dist is not None else ""))

	if distances:
		mean_pix = float(np.mean(distances))
		median_pix = float(np.median(distances))
		print(f"Processed {len(distances)} pairs. Mean pixel distance: {mean_pix:.2f}, Median: {median_pix:.2f}")
		if pixels_to_m is not None:
			print(f"Mean distance in metres: {mean_pix * pixels_to_m:.4f}")
	else:
		print("No distances computed")

	if out_csv:
		with open(out_csv, 'w', newline='') as f:
			writer = csv.writer(f)
			header = ['file', 'pred_x', 'pred_y', 'gt_x', 'gt_y', 'pixel_distance', 'metre_distance']
			writer.writerow(header)
			for r in rows:
				# r may be (basename, None, None, None, None) for missing points
				if len(r) == 5:
					writer.writerow([r[0], None, None, None, None, None, None])
				else:
					writer.writerow(r)
		print(f"Wrote results to {out_csv}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute pickup point distances between predicted and GT masks')
	parser.add_argument('--pred_dir', type=str, default=PRED_MASKS_DIR)
	parser.add_argument('--gt_dir', type=str, default=GT_MASKS_DIR)
	parser.add_argument('--out_csv', type=str, default=None)
	parser.add_argument('--pixels_to_m', type=float, default=None, help='Multiply pixel distance by this to get metres')
	parser.add_argument('--recursive', action='store_true')
	args = parser.parse_args()
	main(args.pred_dir, args.gt_dir, args.out_csv, args.pixels_to_m, args.recursive)

